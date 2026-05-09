package shapekernel

import (
	"fmt"
	"math"
	"strings"
)

// Clip intersects a solid or SDF with an axis-aligned box.
//
// Bounds may use +/-Inf to leave one side unconstrained.
func Clip(k ShapeKernel, minVec, maxVec Vector) ShapeKernel {
	if minVec.Dim() != k.Kind.Dim() || maxVec.Dim() != k.Kind.Dim() {
		panic("clip bounds dimension does not match kernel dimension")
	}
	switch k.Kind {
	case Solid2D, Solid3D:
		return clipSolid(k, minVec, maxVec)
	case SDF2D, SDF3D:
		return clipSDF(k, minVec, maxVec)
	default:
		panic("Clip requires a solid or SDF kernel")
	}
}

// UnionSDFs takes the union of one or more SDFs using max().
func UnionSDFs(sdfs []ShapeKernel) ShapeKernel {
	return sdfBooleanOp(sdfs, "max", "union")
}

// IntersectSDFs takes the intersection of one or more SDFs using min().
func IntersectSDFs(sdfs []ShapeKernel) ShapeKernel {
	return sdfBooleanOp(sdfs, "min", "intersection")
}

// UnionSolids takes the union of one or more solids.
func UnionSolids(solids []ShapeKernel) ShapeKernel {
	return solidBooleanOp(solids, "||", "union")
}

// IntersectSolids takes the intersection of one or more solids.
func IntersectSolids(solids []ShapeKernel) ShapeKernel {
	return solidBooleanOp(solids, "&&", "intersection")
}

// SubtractSolid subtracts negative from positive.
func SubtractSolid(positive, negative ShapeKernel) ShapeKernel {
	if positive.Kind != negative.Kind {
		panic("mismatching shape kinds")
	}
	if positive.Kind != Solid2D && positive.Kind != Solid3D {
		panic("expected solid kernels")
	}
	k := positive
	nextK := ShiftIDs(negative, k.IDs)
	k.IDs = nextK.IDs
	k.Buffers = append(append([]Buffer{}, k.Buffers...), nextK.Buffers...)
	k.Code += "\n" + nextK.Code
	fnName := genFunctionID(&k.IDs, "subtract_solid")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> bool {
				return %s(p) && !%s(p);
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.EntrypointName,
		nextK.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}

// SubtractSDF subtracts negative from positive using min(a, -b).
func SubtractSDF(positive, negative ShapeKernel) ShapeKernel {
	if positive.Kind != negative.Kind {
		panic("mismatching shape kinds")
	}
	if positive.Kind != SDF2D && positive.Kind != SDF3D {
		panic("expected SDF kernels")
	}
	k := positive
	nextK := ShiftIDs(negative, k.IDs)
	k.IDs = nextK.IDs
	k.Buffers = append(append([]Buffer{}, k.Buffers...), nextK.Buffers...)
	k.Code += "\n" + nextK.Code
	fnName := genFunctionID(&k.IDs, "subtract_sdf")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> f32 {
				return min(%s(p), -%s(p));
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.EntrypointName,
		nextK.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}

func solidBooleanOp(solids []ShapeKernel, op, name string) ShapeKernel {
	if len(solids) == 0 {
		panic("expected at least one solid")
	} else if len(solids) == 1 {
		return solids[0]
	}

	for i := 1; i < len(solids); i++ {
		if solids[i].Kind != solids[0].Kind {
			panic("mismatching shape kinds")
		}
	}

	k := solids[0]
	k.Buffers = append([]Buffer{}, k.Buffers...)
	orCode := []string{fmt.Sprintf("%s(p)", k.EntrypointName)}
	for i := 1; i < len(solids); i++ {
		nextK := ShiftIDs(solids[i], k.IDs)
		k.IDs = nextK.IDs
		k.Buffers = append(k.Buffers, nextK.Buffers...)
		k.Code += "\n" + nextK.Code
		orCode = append(orCode, fmt.Sprintf("%s(p)", nextK.EntrypointName))
	}

	fnName := genFunctionID(&k.IDs, name+"_solid")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> bool {
				return %s;
			}
		`),
		fnName,
		k.Kind.ArgType(),
		strings.Join(orCode, " "+op+" "),
	)
	k.EntrypointName = fnName
	return k
}

func clipSolid(k ShapeKernel, minVec, maxVec Vector) ShapeKernel {
	conditions := clipConditions(k.Kind.Dim(), minVec, maxVec)
	if len(conditions) == 0 {
		return k
	}
	fnName := genFunctionID(&k.IDs, "clip_solid")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> bool {
				return %s(p) && (%s);
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.EntrypointName,
		strings.Join(conditions, " && "),
	)
	k.EntrypointName = fnName
	return k
}

func clipSDF(k ShapeKernel, minVec, maxVec Vector) ShapeKernel {
	clipFieldCode, clipFieldName, ok := clipFieldKernel(k.IDs, k.Kind, minVec, maxVec)
	if !ok {
		return k
	}
	k.IDs = clipFieldCode.IDs
	k.Code += "\n" + clipFieldCode.Code
	fnName := genFunctionID(&k.IDs, "clip_sdf")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> f32 {
				return min(%s(p), %s(p));
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.EntrypointName,
		clipFieldName,
	)
	k.EntrypointName = fnName
	return k
}

func clipFieldKernel(ids IDTracker, kind ShapeKind, minVec, maxVec Vector) (ShapeKernel, string, bool) {
	dim := kind.Dim()
	conditions := clipConditions(dim, minVec, maxVec)
	if len(conditions) == 0 {
		return ShapeKernel{IDs: ids}, "", false
	}

	var outsideLets []string
	var outsideTerms []string
	var insideTerms []string
	outsideIdx := 0
	for i := 0; i < dim; i++ {
		component := vectorComponentName(i)
		minVal, maxVal := clipBoundsAt(minVec, maxVec, i)
		if !math.IsInf(float64(minVal), -1) {
			name := fmt.Sprintf("outside_%d", outsideIdx)
			outsideIdx++
			outsideLets = append(outsideLets, fmt.Sprintf("let %s = max(%f - p.%s, 0.0);", name, minVal, component))
			outsideTerms = append(outsideTerms, name)
			insideTerms = append(insideTerms, fmt.Sprintf("p.%s - %f", component, minVal))
		}
		if !math.IsInf(float64(maxVal), 1) {
			name := fmt.Sprintf("outside_%d", outsideIdx)
			outsideIdx++
			outsideLets = append(outsideLets, fmt.Sprintf("let %s = max(p.%s - %f, 0.0);", name, component, maxVal))
			outsideTerms = append(outsideTerms, name)
			insideTerms = append(insideTerms, fmt.Sprintf("%f - p.%s", maxVal, component))
		}
	}

	outsideExpr := outsideTerms[0]
	if len(outsideTerms) > 1 {
		squaredTerms := make([]string, len(outsideTerms))
		for i, term := range outsideTerms {
			squaredTerms[i] = fmt.Sprintf("%s * %s", term, term)
		}
		outsideExpr = fmt.Sprintf("sqrt(%s)", strings.Join(squaredTerms, " + "))
	}

	insideExpr := insideTerms[0]
	for _, term := range insideTerms[1:] {
		insideExpr = fmt.Sprintf("min(%s, %s)", insideExpr, term)
	}

	entrypointName := genFunctionID(&ids, "clip_field")
	return ShapeKernel{
		Kind: kind,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: %s) -> f32 {
					%s
					if (%s) {
						return %s;
					}
					return -(%s);
				}
			`),
			entrypointName,
			kind.ArgType(),
			strings.Join(outsideLets, "\n\t"),
			strings.Join(conditions, " && "),
			insideExpr,
			outsideExpr,
		),
		EntrypointName: entrypointName,
	}, entrypointName, true
}

func clipConditions(dim int, minVec, maxVec Vector) []string {
	var conditions []string
	for i := 0; i < dim; i++ {
		component := vectorComponentName(i)
		minVal, maxVal := clipBoundsAt(minVec, maxVec, i)
		if minVal > maxVal {
			panic("invalid clip bounds")
		}
		if !math.IsInf(float64(minVal), -1) {
			conditions = append(conditions, fmt.Sprintf("p.%s >= %f", component, minVal))
		}
		if !math.IsInf(float64(maxVal), 1) {
			conditions = append(conditions, fmt.Sprintf("p.%s <= %f", component, maxVal))
		}
	}
	return conditions
}

func clipBoundsAt(minVec, maxVec Vector, i int) (float32, float32) {
	minVal := minVec.At(i)
	maxVal := maxVec.At(i)
	if math.IsNaN(float64(minVal)) || math.IsNaN(float64(maxVal)) {
		panic("clip bounds cannot be NaN")
	}
	return minVal, maxVal
}

func vectorComponentName(i int) string {
	switch i {
	case 0:
		return "x"
	case 1:
		return "y"
	case 2:
		return "z"
	default:
		panic("unsupported vector dimension")
	}
}

func sdfBooleanOp(sdfs []ShapeKernel, op, name string) ShapeKernel {
	if len(sdfs) == 0 {
		panic("expected at least one SDF")
	} else if len(sdfs) == 1 {
		return sdfs[0]
	}

	for i := 1; i < len(sdfs); i++ {
		if sdfs[i].Kind != sdfs[0].Kind {
			panic("mismatching shape kinds")
		}
	}
	if sdfs[0].Kind != SDF2D && sdfs[0].Kind != SDF3D {
		panic("expected SDF kernels")
	}

	k := sdfs[0]
	k.Buffers = append([]Buffer{}, k.Buffers...)
	callCode := []string{fmt.Sprintf("%s(p)", k.EntrypointName)}
	for i := 1; i < len(sdfs); i++ {
		nextK := ShiftIDs(sdfs[i], k.IDs)
		k.IDs = nextK.IDs
		k.Buffers = append(k.Buffers, nextK.Buffers...)
		k.Code += "\n" + nextK.Code
		callCode = append(callCode, fmt.Sprintf("%s(p)", nextK.EntrypointName))
	}

	fnName := genFunctionID(&k.IDs, name+"_sdf")
	valueExpr := callCode[0]
	for _, call := range callCode[1:] {
		valueExpr = fmt.Sprintf("%s(%s, %s)", op, valueExpr, call)
	}
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> f32 {
				return %s;
			}
		`),
		fnName,
		k.Kind.ArgType(),
		valueExpr,
	)
	k.EntrypointName = fnName
	return k
}
