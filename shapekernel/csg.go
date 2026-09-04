package shapekernel

import (
	"math"
	"strconv"
	"strings"
)

// Clip intersects a solid or SDF with an axis-aligned box.
//
// Bounds may use +/-Inf to leave one side unconstrained.
func Clip(n Numerics, k ShapeKernel, minVec, maxVec Vector) ShapeKernel {
	if minVec.Dim() != k.Kind.Dim() || maxVec.Dim() != k.Kind.Dim() {
		panic("clip bounds dimension does not match kernel dimension")
	}
	switch k.Kind {
	case Solid2D, Solid3D:
		return clipSolid(n, k, minVec, maxVec)
	case SDF2D, SDF3D:
		return clipSDF(n, k, minVec, maxVec)
	default:
		panic("Clip requires a solid or SDF kernel")
	}
}

// UnionSDFs takes the union of one or more SDFs using max().
func UnionSDFs(n Numerics, sdfs []ShapeKernel) ShapeKernel {
	return sdfBooleanOp(n, sdfs, n.Symbols.Max, "union")
}

// IntersectSDFs takes the intersection of one or more SDFs using min().
func IntersectSDFs(n Numerics, sdfs []ShapeKernel) ShapeKernel {
	return sdfBooleanOp(n, sdfs, n.Symbols.Min, "intersection")
}

// UnionSolids takes the union of one or more solids.
func UnionSolids(n Numerics, solids []ShapeKernel) ShapeKernel {
	return solidBooleanOp(n, solids, "||", "union")
}

// IntersectSolids takes the intersection of one or more solids.
func IntersectSolids(n Numerics, solids []ShapeKernel) ShapeKernel {
	return solidBooleanOp(n, solids, "&&", "intersection")
}

// SubtractSolid subtracts negative from positive.
func SubtractSolid(n Numerics, positive, negative ShapeKernel) ShapeKernel {
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
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> bool {
				return {{.Positive}}(p) && !{{.Negative}}(p);
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"Positive", k.EntrypointName,
		"Negative", nextK.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}

// SubtractSDF subtracts negative from positive using min(a, -b).
func SubtractSDF(n Numerics, positive, negative ShapeKernel) ShapeKernel {
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
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				return {{.N.Min}}({{.Positive}}(p), {{.N.Sub}}({{.N.Zero}}, {{.Negative}}(p)));
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"ReturnType", k.Kind.ReturnType(n),
		"Positive", k.EntrypointName,
		"Negative", nextK.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}

func solidBooleanOp(n Numerics, solids []ShapeKernel, op, name string) ShapeKernel {
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
	mutationCode := []string{WGSL("var value = {{.Fn}}(p);", "Fn", k.EntrypointName)}
	for i := 1; i < len(solids); i++ {
		nextK := ShiftIDs(solids[i], k.IDs)
		k.IDs = nextK.IDs
		k.Buffers = append(k.Buffers, nextK.Buffers...)
		k.Code += "\n" + nextK.Code
		mutationCode = append(mutationCode, WGSL("value = value {{.Op}} {{.Fn}}(p);", "Op", op, "Fn", nextK.EntrypointName))
	}

	fnName := genFunctionID(&k.IDs, name+"_solid")
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> bool {
				{{.MutationCode}}
				return value;
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"MutationCode", strings.Join(mutationCode, "\n"),
	)
	k.EntrypointName = fnName
	return k
}

func clipSolid(n Numerics, k ShapeKernel, minVec, maxVec Vector) ShapeKernel {
	conditions := clipConditions(n, k.Kind.Dim(), minVec, maxVec, "p")
	if len(conditions) == 0 {
		return k
	}
	fnName := genFunctionID(&k.IDs, "clip_solid")
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> bool {
				return {{.Inner}}(p) && ({{.Conditions}});
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"Inner", k.EntrypointName,
		"Conditions", strings.Join(conditions, " && "),
	)
	k.EntrypointName = fnName
	return k
}

func clipSDF(n Numerics, k ShapeKernel, minVec, maxVec Vector) ShapeKernel {
	clipFieldCode, clipFieldName, ok := clipFieldKernel(n, k.IDs, k.Kind, minVec, maxVec)
	if !ok {
		return k
	}
	k.IDs = clipFieldCode.IDs
	k.Code += "\n" + clipFieldCode.Code
	fnName := genFunctionID(&k.IDs, "clip_sdf")
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				return {{.N.Min}}({{.Inner}}(p), {{.ClipField}}(p));
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"ReturnType", k.Kind.ReturnType(n),
		"Inner", k.EntrypointName,
		"ClipField", clipFieldName,
	)
	k.EntrypointName = fnName
	return k
}

func clipFieldKernel(n Numerics, ids IDTracker, kind ShapeKind, minVec, maxVec Vector) (ShapeKernel, string, bool) {
	dim := kind.Dim()
	conditions := clipConditions(n, dim, minVec, maxVec, "p")
	if len(conditions) == 0 {
		return ShapeKernel{IDs: ids}, "", false
	}

	var outsideLets []string
	var outsideTerms []string
	var insideTerms []string
	outsideIdx := 0
	for i := 0; i < dim; i++ {
		minVal, maxVal := clipBoundsAt(minVec, maxVec, i)
		componentExpr := vectorComponentExpr(n, dim, "p", i)
		if !math.IsInf(minVal, -1) {
			name := "outside_" + strconv.Itoa(outsideIdx)
			outsideIdx++
			outsideLets = append(outsideLets, WGSL("let {{.Name}} = {{.N.Max}}({{.N.Sub}}({{.Min}}, {{.Component}}), {{.N.Zero}});", "N", n.Symbols, "Name", name, "Min", n.Literal(minVal), "Component", componentExpr))
			outsideTerms = append(outsideTerms, name)
			insideTerms = append(insideTerms, WGSL("{{.N.Sub}}({{.Component}}, {{.Min}})", "N", n.Symbols, "Component", componentExpr, "Min", n.Literal(minVal)))
		}
		if !math.IsInf(maxVal, 1) {
			name := "outside_" + strconv.Itoa(outsideIdx)
			outsideIdx++
			outsideLets = append(outsideLets, WGSL("let {{.Name}} = {{.N.Max}}({{.N.Sub}}({{.Component}}, {{.Max}}), {{.N.Zero}});", "N", n.Symbols, "Name", name, "Component", componentExpr, "Max", n.Literal(maxVal)))
			outsideTerms = append(outsideTerms, name)
			insideTerms = append(insideTerms, WGSL("{{.N.Sub}}({{.Max}}, {{.Component}})", "N", n.Symbols, "Max", n.Literal(maxVal), "Component", componentExpr))
		}
	}

	outsideExpr := outsideTerms[0]
	if len(outsideTerms) > 1 {
		squaredTerms := make([]string, len(outsideTerms))
		for i, term := range outsideTerms {
			squaredTerms[i] = WGSL("{{.N.Mul}}({{.Term}}, {{.Term}})", "N", n.Symbols, "Term", term)
		}
		sumExpr := squaredTerms[0]
		for _, term := range squaredTerms[1:] {
			sumExpr = WGSL("{{.N.Add}}({{.Left}}, {{.Right}})", "N", n.Symbols, "Left", sumExpr, "Right", term)
		}
		outsideExpr = WGSL("{{.N.Sqrt}}({{.Expr}})", "N", n.Symbols, "Expr", sumExpr)
	}

	insideExpr := insideTerms[0]
	for _, term := range insideTerms[1:] {
		insideExpr = WGSL("{{.N.Min}}({{.Left}}, {{.Right}})", "N", n.Symbols, "Left", insideExpr, "Right", term)
	}

	entrypointName := genFunctionID(&ids, "clip_field")
	return ShapeKernel{
		Kind: kind,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
					{{.OutsideLets}}
					if ({{.Conditions}}) {
						return {{.InsideExpr}};
					}
					return {{.N.Sub}}({{.N.Zero}}, {{.OutsideExpr}});
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"ArgType", kind.ArgType(n),
			"ReturnType", kind.ReturnType(n),
			"OutsideLets", strings.Join(outsideLets, "\n\t"),
			"Conditions", strings.Join(conditions, " && "),
			"InsideExpr", insideExpr,
			"OutsideExpr", outsideExpr,
		),
		EntrypointName: entrypointName,
	}, entrypointName, true
}

func clipConditions(n Numerics, dim int, minVec, maxVec Vector, pointName string) []string {
	var conditions []string
	for i := 0; i < dim; i++ {
		minVal, maxVal := clipBoundsAt(minVec, maxVec, i)
		if minVal > maxVal {
			panic("invalid clip bounds")
		}
		componentExpr := vectorComponentExpr(n, dim, pointName, i)
		if !math.IsInf(minVal, -1) {
			conditions = append(conditions, WGSL("{{.N.Ge}}({{.Component}}, {{.Min}})", "N", n.Symbols, "Component", componentExpr, "Min", n.Literal(minVal)))
		}
		if !math.IsInf(maxVal, 1) {
			conditions = append(conditions, WGSL("{{.N.Le}}({{.Component}}, {{.Max}})", "N", n.Symbols, "Component", componentExpr, "Max", n.Literal(maxVal)))
		}
	}
	return conditions
}

func clipBoundsAt(minVec, maxVec Vector, i int) (float64, float64) {
	minVal := minVec.At(i)
	maxVal := maxVec.At(i)
	if math.IsNaN(minVal) || math.IsNaN(maxVal) {
		panic("clip bounds cannot be NaN")
	}
	return minVal, maxVal
}

func vectorComponentExpr(n Numerics, dim int, vectorName string, i int) string {
	if dim == 2 {
		switch i {
		case 0:
			return WGSL("{{.N.Get2X}}({{.Vector}})", "N", n.Symbols, "Vector", vectorName)
		case 1:
			return WGSL("{{.N.Get2Y}}({{.Vector}})", "N", n.Symbols, "Vector", vectorName)
		}
	} else if dim == 3 {
		switch i {
		case 0:
			return WGSL("{{.N.Get3X}}({{.Vector}})", "N", n.Symbols, "Vector", vectorName)
		case 1:
			return WGSL("{{.N.Get3Y}}({{.Vector}})", "N", n.Symbols, "Vector", vectorName)
		case 2:
			return WGSL("{{.N.Get3Z}}({{.Vector}})", "N", n.Symbols, "Vector", vectorName)
		}
	}
	panic("unsupported vector dimension")
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

func sdfBooleanOp(n Numerics, sdfs []ShapeKernel, op, name string) ShapeKernel {
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
	mutationCode := []string{WGSL("var value = {{.Fn}}(p);", "Fn", k.EntrypointName)}
	for i := 1; i < len(sdfs); i++ {
		nextK := ShiftIDs(sdfs[i], k.IDs)
		k.IDs = nextK.IDs
		k.Buffers = append(k.Buffers, nextK.Buffers...)
		k.Code += "\n" + nextK.Code
		mutationCode = append(mutationCode, WGSL("value = {{.Op}}(value, {{.Fn}}(p));", "Op", op, "Fn", nextK.EntrypointName))
	}

	fnName := genFunctionID(&k.IDs, name+"_sdf")
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				{{.MutationCode}}
				return value;
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"ReturnType", k.Kind.ReturnType(n),
		"MutationCode", strings.Join(mutationCode, "\n"),
	)
	k.EntrypointName = fnName
	return k
}
