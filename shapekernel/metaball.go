package shapekernel

import "strings"

func InversePowerMetaballFalloffFunc(power float32) ShapeKernel {
	if power <= 0 {
		panic("power must be positive")
	}
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "inverse_power_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(r: f32) -> f32 {
					if (r <= 0.0) {
						return 1e30;
					}
					return pow(r, -{{.Power}});
				}
			`, "Entrypoint", entrypointName, "Power", power),
		EntrypointName: entrypointName,
	}
}

func LinearMetaballFalloffFunc() ShapeKernel {
	return InversePowerMetaballFalloffFunc(1)
}

func QuadraticMetaballFalloffFunc() ShapeKernel {
	return InversePowerMetaballFalloffFunc(2)
}

func CubicMetaballFalloffFunc() ShapeKernel {
	return InversePowerMetaballFalloffFunc(3)
}

func QuarticMetaballFalloffFunc() ShapeKernel {
	return InversePowerMetaballFalloffFunc(4)
}

func QuinticMetaballFalloffFunc() ShapeKernel {
	return InversePowerMetaballFalloffFunc(5)
}

func ExponentialMetaballFalloffFunc() ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "exponential_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(r: f32) -> f32 {
					if (r <= 0.0) {
						return 1.0;
					}
					return exp(-r);
				}
			`, "Entrypoint", entrypointName),
		EntrypointName: entrypointName,
	}
}

func GaussianMetaballFalloffFunc() ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "gaussian_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(r: f32) -> f32 {
					if (r <= 0.0) {
						return 1.0;
					}
					return exp(-(r * r));
				}
			`, "Entrypoint", entrypointName),
		EntrypointName: entrypointName,
	}
}

func WyvillMetaballFalloffFunc(d float32) ShapeKernel {
	if d <= 0 {
		panic("d must be positive")
	}
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "wyvill_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(r: f32) -> f32 {
					if (r < 0.0) {
						return 1e30;
					}
					if (r >= {{.Radius}}) {
						return 0.0;
					}
					let ratio2 = (r * r) / ({{.Radius}} * {{.Radius}});
					let value = 1.0 - ratio2;
					return value * value;
				}
			`, "Entrypoint", entrypointName, "Radius", d),
		EntrypointName: entrypointName,
	}
}

func SDFToMetaball(k ShapeKernel) ShapeKernel {
	var metaballKind ShapeKind
	switch k.Kind {
	case SDF2D:
		metaballKind = Metaball2D
	case SDF3D:
		metaballKind = Metaball3D
	default:
		panic("expected SDF shape kernel")
	}

	fnName := genFunctionID(&k.IDs, "sdf_to_metaball")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> f32 {
				return -{{.Inner}}(p);
			}
		`, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(), "Inner", k.EntrypointName)
	k.Kind = metaballKind
	k.EntrypointName = fnName
	return k
}

func MetaballSolid(falloff ShapeKernel, radiusThreshold float32, metaballs ...ShapeKernel) ShapeKernel {
	weights := make([]float32, len(metaballs))
	for i := range weights {
		weights[i] = 1
	}
	return WeightedMetaballSolid(falloff, radiusThreshold, metaballs, weights)
}

func WeightedMetaballSolid(
	falloff ShapeKernel,
	radiusThreshold float32,
	metaballs []ShapeKernel,
	weights []float32,
) ShapeKernel {
	if len(metaballs) == 0 {
		panic("expected at least one metaball")
	}
	if len(metaballs) != len(weights) {
		panic("number of metaballs must match number of weights")
	}
	if falloff.Kind != FalloffFunc {
		panic("expected falloff function kernel")
	}

	kind := metaballs[0].Kind
	if kind != Metaball2D && kind != Metaball3D {
		panic("expected metaball kernel")
	}
	for i := 1; i < len(metaballs); i++ {
		if metaballs[i].Kind != kind {
			panic("mismatching metaball kinds")
		}
	}

	k := metaballs[0]
	k.Buffers = append([]Buffer{}, k.Buffers...)
	metaballCalls := []string{WGSL("{{.Fn}}(p)", "Fn", k.EntrypointName)}
	for i := 1; i < len(metaballs); i++ {
		nextK := ShiftIDs(metaballs[i], k.IDs)
		k.IDs = nextK.IDs
		k.Buffers = append(k.Buffers, nextK.Buffers...)
		k.Code += "\n" + nextK.Code
		metaballCalls = append(metaballCalls, WGSL("{{.Fn}}(p)", "Fn", nextK.EntrypointName))
	}

	falloff = ShiftIDs(falloff, k.IDs)
	k.IDs = falloff.IDs
	k.Buffers = append(k.Buffers, falloff.Buffers...)
	k.Code += "\n" + falloff.Code
	sumCode := make([]string, len(metaballCalls))
	for i, call := range metaballCalls {
		sumCode[i] = WGSL("{{.Weight}} * {{.Falloff}}({{.Call}})",
			"Weight", weights[i],
			"Falloff", falloff.EntrypointName,
			"Call", call,
		)
	}

	solidKind := Solid2D
	if kind == Metaball3D {
		solidKind = Solid3D
	}
	fnName := genFunctionID(&k.IDs, "metaball_solid")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> bool {
				let threshold = {{.Falloff}}({{.RadiusThreshold}});
				let sum = {{.SumExpr}};
				return sum > threshold;
			}
		`,
		"Entrypoint", fnName,
		"ArgType", kind.ArgType(),
		"Falloff", falloff.EntrypointName,
		"RadiusThreshold", radiusThreshold,
		"SumExpr", strings.Join(sumCode, " + "),
	)
	k.Kind = solidKind
	k.EntrypointName = fnName
	return k
}
