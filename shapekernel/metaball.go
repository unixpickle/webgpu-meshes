package shapekernel

func InversePowerMetaballFalloffFunc(n Numerics, power float32) ShapeKernel {
	if power <= 0 {
		panic("power must be positive")
	}
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "inverse_power_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
					fn {{.Entrypoint}}(r: {{.N.Dtype}}) -> {{.N.Dtype}} {
						if ({{.N.Le}}(r, {{.N.Zero}})) {
							return {{.Inf}};
						}
						return {{.N.Pow}}(r, {{.NegPower}});
					}
				`, "N", n.Symbols, "Entrypoint", entrypointName, "Inf", n.Literal(1e30), "NegPower", n.Literal(float64(-power))),
		EntrypointName: entrypointName,
	}
}

func LinearMetaballFalloffFunc(n Numerics) ShapeKernel {
	return InversePowerMetaballFalloffFunc(n, 1)
}

func QuadraticMetaballFalloffFunc(n Numerics) ShapeKernel {
	return InversePowerMetaballFalloffFunc(n, 2)
}

func CubicMetaballFalloffFunc(n Numerics) ShapeKernel {
	return InversePowerMetaballFalloffFunc(n, 3)
}

func QuarticMetaballFalloffFunc(n Numerics) ShapeKernel {
	return InversePowerMetaballFalloffFunc(n, 4)
}

func QuinticMetaballFalloffFunc(n Numerics) ShapeKernel {
	return InversePowerMetaballFalloffFunc(n, 5)
}

func ExponentialMetaballFalloffFunc(n Numerics) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "exponential_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
					fn {{.Entrypoint}}(r: {{.N.Dtype}}) -> {{.N.Dtype}} {
						if ({{.N.Le}}(r, {{.N.Zero}})) {
							return {{.N.One}};
						}
						return {{.N.Exp}}({{.N.Sub}}({{.N.Zero}}, r));
					}
				`, "N", n.Symbols, "Entrypoint", entrypointName),
		EntrypointName: entrypointName,
	}
}

func GaussianMetaballFalloffFunc(n Numerics) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "gaussian_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
					fn {{.Entrypoint}}(r: {{.N.Dtype}}) -> {{.N.Dtype}} {
						if ({{.N.Le}}(r, {{.N.Zero}})) {
							return {{.N.One}};
						}
						return {{.N.Exp}}({{.N.Sub}}({{.N.Zero}}, {{.N.Mul}}(r, r)));
					}
				`, "N", n.Symbols, "Entrypoint", entrypointName),
		EntrypointName: entrypointName,
	}
}

func WyvillMetaballFalloffFunc(n Numerics, d float32) ShapeKernel {
	if d <= 0 {
		panic("d must be positive")
	}
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "wyvill_metaball_falloff")
	return ShapeKernel{
		Kind: FalloffFunc,
		IDs:  ids,
		Code: WGSL(`
					fn {{.Entrypoint}}(r: {{.N.Dtype}}) -> {{.N.Dtype}} {
						if ({{.N.Lt}}(r, {{.N.Zero}})) {
							return {{.Inf}};
						}
						if ({{.N.Ge}}(r, {{.Radius}})) {
							return {{.N.Zero}};
						}
						let ratio2 = {{.N.Div}}({{.N.Mul}}(r, r), {{.N.Mul}}({{.Radius}}, {{.Radius}}));
						let value = {{.N.Sub}}({{.N.One}}, ratio2);
						return {{.N.Mul}}(value, value);
					}
				`, "N", n.Symbols, "Entrypoint", entrypointName, "Radius", n.Literal(float64(d)), "Inf", n.Literal(1e30)),
		EntrypointName: entrypointName,
	}
}

func SDFToMetaball(n Numerics, k ShapeKernel) ShapeKernel {
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
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
				return {{.N.Sub}}({{.N.Zero}}, {{.Inner}}(p));
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "ArgType", k.Kind.ArgType(n), "ReturnType", k.Kind.ReturnType(n), "Inner", k.EntrypointName)
	k.Kind = metaballKind
	k.EntrypointName = fnName
	return k
}

func MetaballSolid(n Numerics, falloff ShapeKernel, radiusThreshold float32, metaballs ...ShapeKernel) ShapeKernel {
	weights := make([]float32, len(metaballs))
	for i := range weights {
		weights[i] = 1
	}
	return WeightedMetaballSolid(n, falloff, radiusThreshold, metaballs, weights)
}

func WeightedMetaballSolid(
	n Numerics,
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
		sumCode[i] = WGSL("{{.N.Mul}}({{.Weight}}, {{.Falloff}}({{.Call}}))",
			"N", n.Symbols,
			"Weight", n.Literal(float64(weights[i])),
			"Falloff", falloff.EntrypointName,
			"Call", call,
		)
	}
	sumExpr := sumCode[0]
	for _, term := range sumCode[1:] {
		sumExpr = WGSL("{{.N.Add}}({{.Left}}, {{.Right}})", "N", n.Symbols, "Left", sumExpr, "Right", term)
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
					return {{.N.Gt}}(sum, threshold);
				}
			`,
		"Entrypoint", fnName,
		"ArgType", kind.ArgType(n),
		"N", n.Symbols,
		"Falloff", falloff.EntrypointName,
		"RadiusThreshold", n.Literal(float64(radiusThreshold)),
		"SumExpr", sumExpr,
	)
	k.Kind = solidKind
	k.EntrypointName = fnName
	return k
}
