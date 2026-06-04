package shapekernel

import "math"

func unitAxis2D(axis Vec2) Vec2 {
	axisNorm := math.Sqrt(float64(axis[0]*axis[0] + axis[1]*axis[1]))
	if axisNorm == 0 {
		panic("expected a non-zero axis")
	}
	return Vec2{
		float64(axis[0]) / axisNorm,
		float64(axis[1]) / axisNorm,
	}
}

func unitAxis3D(axis Vec3) Vec3 {
	axisNorm := math.Sqrt(float64(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]))
	if axisNorm == 0 {
		panic("expected a non-zero axis")
	}
	return Vec3{
		float64(axis[0]) / axisNorm,
		float64(axis[1]) / axisNorm,
		float64(axis[2]) / axisNorm,
	}
}

func absScaleFactor(kind ShapeKind, scales Vector) float64 {
	if scales.Dim() != kind.Dim() {
		panic("scale dimension does not match kernel dimension")
	}
	abs0 := math.Abs(float64(scales.At(0)))
	if abs0 == 0 {
		panic("scale components must be non-zero")
	}
	for i := 1; i < scales.Dim(); i++ {
		absValue := math.Abs(float64(scales.At(i)))
		if absValue == 0 {
			panic("scale components must be non-zero")
		}
		if kind == SDF2D || kind == SDF3D {
			maxScale := math.Max(abs0, absValue)
			if math.Abs(absValue-abs0) > 1e-6*math.Max(1, maxScale) {
				panic("SDF kernels require equal absolute scale on every axis")
			}
		}
	}
	return abs0
}

func Rotate2D(n Numerics, k ShapeKernel, angle float32) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 2 {
		panic("Rotate2D requires a 2D non-falloff kernel")
	}
	fnName := genFunctionID(&k.IDs, "rotate2d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.ArgType}}) -> {{.ReturnType}} {
				let p = {{.N.AsFloat2}}(pRaw);
				let cosA = cos({{.Angle}});
				let sinA = sin({{.Angle}});
				let newP = vec2<f32>(
					cosA * p.x + sinA * p.y,
					-sinA * p.x + cosA * p.y
				);
				return {{.Inner}}({{.N.Make2}}({{.N.FromFloat}}(newP.x), {{.N.FromFloat}}(newP.y)));
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(n), "ReturnType", k.Kind.ReturnType(n), "Angle", angle, "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Rotate3D(n Numerics, k ShapeKernel, axis Vec3, angle float32) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 3 {
		panic("Rotate3D requires a 3D non-falloff kernel")
	}
	unitAxis := unitAxis3D(axis)
	fnName := genFunctionID(&k.IDs, "rotate3d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.ArgType}}) -> {{.ReturnType}} {
				let p = {{.N.AsFloat3}}(pRaw);
				let axis = {{.N.AsFloat3}}({{.Axis}});
				let cosA = cos({{.Angle}});
				let sinA = sin({{.Angle}});
				let newP = p * cosA - cross(axis, p) * sinA + axis * dot(axis, p) * (1.0 - cosA);
				return {{.Inner}}({{.N.Make3}}({{.N.FromFloat}}(newP.x), {{.N.FromFloat}}(newP.y), {{.N.FromFloat}}(newP.z)));
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(n), "ReturnType", k.Kind.ReturnType(n), "Axis", unitAxis.WebGPUVec(n), "Angle", angle, "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Mirror2D(n Numerics, k ShapeKernel, axis Vec2) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 2 {
		panic("Mirror2D requires a 2D non-falloff kernel")
	}
	unitAxis := unitAxis2D(axis)
	fnName := genFunctionID(&k.IDs, "mirror2d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.ArgType}}) -> {{.ReturnType}} {
				let p = {{.N.AsFloat2}}(pRaw);
				let axis = {{.N.AsFloat2}}({{.Axis}});
				let newP = p - 2.0 * dot(axis, p) * axis;
				return {{.Inner}}({{.N.Make2}}({{.N.FromFloat}}(newP.x), {{.N.FromFloat}}(newP.y)));
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(n), "ReturnType", k.Kind.ReturnType(n), "Axis", unitAxis.WebGPUVec(n), "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Mirror3D(n Numerics, k ShapeKernel, axis Vec3) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 3 {
		panic("Mirror3D requires a 3D non-falloff kernel")
	}
	unitAxis := unitAxis3D(axis)
	fnName := genFunctionID(&k.IDs, "mirror3d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.ArgType}}) -> {{.ReturnType}} {
				let p = {{.N.AsFloat3}}(pRaw);
				let axis = {{.N.AsFloat3}}({{.Axis}});
				let newP = p - 2.0 * dot(axis, p) * axis;
				return {{.Inner}}({{.N.Make3}}({{.N.FromFloat}}(newP.x), {{.N.FromFloat}}(newP.y), {{.N.FromFloat}}(newP.z)));
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(n), "ReturnType", k.Kind.ReturnType(n), "Axis", unitAxis.WebGPUVec(n), "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Translate(n Numerics, k ShapeKernel, offset Vector) ShapeKernel {
	if k.Kind == FalloffFunc {
		panic("cannot translate falloff functions")
	}
	fnName := genFunctionID(&k.IDs, "translate")
	diffOp := n.Symbols.Sub2
	if k.Kind.Dim() == 3 {
		diffOp = n.Symbols.Sub3
	}
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				let newP = {{.DiffOp}}(p, {{.Offset}});
				return {{.Inner}}(newP);
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"ReturnType", k.Kind.ReturnType(n),
		"Offset", offset.WebGPUVec(n),
		"Inner", k.EntrypointName,
		"DiffOp", diffOp,
	)
	k.EntrypointName = fnName
	return k
}

// InsetSDF offsets an SDF inward by subtracting inset from its value.
func InsetSDF(n Numerics, k ShapeKernel, inset float32) ShapeKernel {
	return offsetSDF(n, k, -inset, "inset")
}

// OutsetSDF offsets an SDF outward by adding outset to its value.
func OutsetSDF(n Numerics, k ShapeKernel, outset float32) ShapeKernel {
	return offsetSDF(n, k, outset, "outset")
}

func Scale(n Numerics, k ShapeKernel, scales Vector) ShapeKernel {
	if k.Kind == FalloffFunc {
		panic("cannot scale falloff functions")
	}
	absScale := absScaleFactor(k.Kind, scales)
	resultExpr := "inner"
	if k.Kind == SDF2D || k.Kind == SDF3D {
		resultExpr = WGSL("{{.N.Mul}}(inner, {{.AbsScale}})", "N", n.Symbols, "AbsScale", n.Literal(absScale))
	}
	divOp := n.Symbols.Div2
	if k.Kind.Dim() == 3 {
		divOp = n.Symbols.Div3
	}
	fnName := genFunctionID(&k.IDs, "scale")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				let newP = {{.DivOp}}(p, {{.Scales}});
				let inner = {{.Inner}}(newP);
				return {{.ResultExpr}};
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(n),
		"ReturnType", k.Kind.ReturnType(n),
		"Scales", scales.WebGPUVec(n),
		"Inner", k.EntrypointName,
		"DivOp", divOp,
		"ResultExpr", resultExpr,
	)
	k.EntrypointName = fnName
	return k
}

func offsetSDF(n Numerics, k ShapeKernel, offset float32, name string) ShapeKernel {
	if k.Kind != SDF2D && k.Kind != SDF3D {
		panic("expected SDF kernel")
	}
	fnName := genFunctionID(&k.IDs, name+"_sdf")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				return {{.N.Add}}({{.Inner}}(p), {{.Offset}});
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(n), "ReturnType", k.Kind.ReturnType(n), "Inner", k.EntrypointName, "Offset", n.Literal(float64(offset)))
	k.EntrypointName = fnName
	return k
}
