package shapekernel

import "math"

func unitAxis2D(axis Vec2) Vec2 {
	axisNorm := math.Sqrt(float64(axis[0]*axis[0] + axis[1]*axis[1]))
	if axisNorm == 0 {
		panic("expected a non-zero axis")
	}
	return Vec2{
		float32(float64(axis[0]) / axisNorm),
		float32(float64(axis[1]) / axisNorm),
	}
}

func unitAxis3D(axis Vec3) Vec3 {
	axisNorm := math.Sqrt(float64(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]))
	if axisNorm == 0 {
		panic("expected a non-zero axis")
	}
	return Vec3{
		float32(float64(axis[0]) / axisNorm),
		float32(float64(axis[1]) / axisNorm),
		float32(float64(axis[2]) / axisNorm),
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

func Rotate2D(k ShapeKernel, angle float32) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 2 {
		panic("Rotate2D requires a 2D non-falloff kernel")
	}
	fnName := genFunctionID(&k.IDs, "rotate2d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: vec2<f32>) -> {{.ReturnType}} {
				let cosA = cos({{.Angle}});
				let sinA = sin({{.Angle}});
				let newP = vec2<f32>(
					cosA * p.x + sinA * p.y,
					-sinA * p.x + cosA * p.y
				);
				return {{.Inner}}(newP);
			}
		`, "Entrypoint", fnName, "ReturnType", k.Kind.ReturnType(), "Angle", angle, "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Rotate3D(k ShapeKernel, axis Vec3, angle float32) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 3 {
		panic("Rotate3D requires a 3D non-falloff kernel")
	}
	unitAxis := unitAxis3D(axis)
	fnName := genFunctionID(&k.IDs, "rotate3d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: vec3<f32>) -> {{.ReturnType}} {
				let axis = {{.Axis}};
				let cosA = cos({{.Angle}});
				let sinA = sin({{.Angle}});
				let newP = p * cosA - cross(axis, p) * sinA + axis * dot(axis, p) * (1.0 - cosA);
				return {{.Inner}}(newP);
			}
		`, "Entrypoint", fnName, "ReturnType", k.Kind.ReturnType(), "Axis", unitAxis.WebGPUVec(), "Angle", angle, "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Mirror2D(k ShapeKernel, axis Vec2) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 2 {
		panic("Mirror2D requires a 2D non-falloff kernel")
	}
	unitAxis := unitAxis2D(axis)
	fnName := genFunctionID(&k.IDs, "mirror2d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: vec2<f32>) -> {{.ReturnType}} {
				let axis = {{.Axis}};
				let newP = p - 2.0 * dot(axis, p) * axis;
				return {{.Inner}}(newP);
			}
		`, "Entrypoint", fnName, "ReturnType", k.Kind.ReturnType(), "Axis", unitAxis.WebGPUVec(), "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Mirror3D(k ShapeKernel, axis Vec3) ShapeKernel {
	if k.Kind == FalloffFunc || k.Kind.Dim() != 3 {
		panic("Mirror3D requires a 3D non-falloff kernel")
	}
	unitAxis := unitAxis3D(axis)
	fnName := genFunctionID(&k.IDs, "mirror3d")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: vec3<f32>) -> {{.ReturnType}} {
				let axis = {{.Axis}};
				let newP = p - 2.0 * dot(axis, p) * axis;
				return {{.Inner}}(newP);
			}
		`, "Entrypoint", fnName, "ReturnType", k.Kind.ReturnType(), "Axis", unitAxis.WebGPUVec(), "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

func Translate(k ShapeKernel, offset Vector) ShapeKernel {
	if k.Kind == FalloffFunc {
		panic("cannot translate falloff functions")
	}
	fnName := genFunctionID(&k.IDs, "translate")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				let newP = p - {{.Offset}};
				return {{.Inner}}(newP);
			}
		`, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(), "ReturnType", k.Kind.ReturnType(), "Offset", offset.WebGPUVec(), "Inner", k.EntrypointName)
	k.EntrypointName = fnName
	return k
}

// InsetSDF offsets an SDF inward by subtracting inset from its value.
func InsetSDF(k ShapeKernel, inset float32) ShapeKernel {
	return offsetSDF(k, -inset, "inset")
}

// OutsetSDF offsets an SDF outward by adding outset to its value.
func OutsetSDF(k ShapeKernel, outset float32) ShapeKernel {
	return offsetSDF(k, outset, "outset")
}

func Scale(k ShapeKernel, scales Vector) ShapeKernel {
	if k.Kind == FalloffFunc {
		panic("cannot scale falloff functions")
	}
	absScale := absScaleFactor(k.Kind, scales)
	scaleCode := ""
	if k.Kind == SDF2D || k.Kind == SDF3D {
		scaleCode = Template(" * {{.AbsScale}}", "AbsScale", absScale)
	}
	fnName := genFunctionID(&k.IDs, "scale")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				let newP = p / {{.Scales}};
				return {{.Inner}}(newP){{.ScaleCode}};
			}
		`,
		"Entrypoint", fnName,
		"ArgType", k.Kind.ArgType(),
		"ReturnType", k.Kind.ReturnType(),
		"Scales", scales.WebGPUVec(),
		"Inner", k.EntrypointName,
		"ScaleCode", scaleCode,
	)
	k.EntrypointName = fnName
	return k
}

func offsetSDF(k ShapeKernel, offset float32, name string) ShapeKernel {
	if k.Kind != SDF2D && k.Kind != SDF3D {
		panic("expected SDF kernel")
	}
	fnName := genFunctionID(&k.IDs, name+"_sdf")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> f32 {
				return {{.Inner}}(p) + {{.Offset}};
			}
		`, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(), "Inner", k.EntrypointName, "Offset", offset)
	k.EntrypointName = fnName
	return k
}
