package shapekernel

import (
	"math"
)

const (
	InsetExtrudeChamfer InsetFunction = "chamfer"
	InsetExtrudeFillet  InsetFunction = "fillet"
)

type InsetFunction string

// LinearExtrudeSolid extends a 2D shape along the Z axis, optionally centered,
// twisted, and scaled from bottom to top.
func LinearExtrudeSolid(n Numerics, k ShapeKernel, height float64, center bool, twist float64, scale Vec2) ShapeKernel {
	switch k.Kind {
	case SDF2D:
		k = SDFToSolid(n, k)
	case Solid2D:
	default:
		panic("expected 2D solid or SDF kernel")
	}

	if height < 0 {
		height = -height
	}
	z0, z1 := linearExtrudeZBounds(height, center)
	fnName := genFunctionID(&k.IDs, "linear_extrude")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
				if ({{.N.Lt}}({{.N.Get3Z}}(p), {{.ZMin}}) || {{.N.Gt}}({{.N.Get3Z}}(p), {{.ZMax}})) {
					return false;
				}

				var t = {{.N.Zero}};
				if ({{.N.Gt}}({{.Height}}, {{.N.Zero}})) {
					t = {{.N.Div}}({{.N.Sub}}({{.N.Get3Z}}(p), {{.ZMin}}), {{.Height}});
				}

				let sx = {{.N.Add}}({{.N.One}}, {{.N.Mul}}(t, {{.N.Sub}}({{.ScaleX}}, {{.N.One}})));
				let sy = {{.N.Add}}({{.N.One}}, {{.N.Mul}}(t, {{.N.Sub}}({{.ScaleY}}, {{.N.One}})));
				if ({{.N.Eq}}(sx, {{.N.Zero}}) || {{.N.Eq}}(sy, {{.N.Zero}})) {
					return false;
				}

				let angle = {{.N.Mul}}({{.Twist}}, t);
				let cosA = {{.N.Cos}}(angle);
				let sinA = {{.N.Sin}}(angle);
				let rx = {{.N.Sub}}({{.N.Mul}}({{.N.Get3X}}(p), cosA), {{.N.Mul}}({{.N.Get3Y}}(p), sinA));
				let ry = {{.N.Add}}({{.N.Mul}}({{.N.Get3X}}(p), sinA), {{.N.Mul}}({{.N.Get3Y}}(p), cosA));
				return {{.Inner}}({{.N.Make2}}({{.N.Div}}(rx, sx), {{.N.Div}}(ry, sy)));
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"ZMin", n.Literal(z0),
		"ZMax", n.Literal(z1),
		"Height", n.Literal(height),
		"ScaleX", n.Literal(scale[0]),
		"ScaleY", n.Literal(scale[1]),
		"Twist", n.Literal(twist),
		"Inner", k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

// LinearExtrudeSDF turns a 2D SDF into a 3D SDF by extruding it along the Z
// axis with a height and optional centering.
func LinearExtrudeSDF(n Numerics, k ShapeKernel, height float64, center bool) ShapeKernel {
	if k.Kind != SDF2D {
		panic("expected 2D SDF kernel")
	}
	if height < 0 {
		height = -height
	}
	z0, z1 := linearExtrudeZBounds(height, center)
	fnName := genFunctionID(&k.IDs, "linear_extrude_sdf")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> {{.N.Dtype}} {
				let p2d = {{.N.Make2}}({{.N.Get3X}}(p), {{.N.Get3Y}}(p));
				let sdf2d = {{.Inner}}(p2d);
				let zDist = {{.N.Min}}({{.N.Abs}}({{.N.Sub}}({{.N.Get3Z}}(p), {{.ZMin}})), {{.N.Abs}}({{.N.Sub}}({{.N.Get3Z}}(p), {{.ZMax}})));
				let insideZ = {{.N.Ge}}({{.N.Get3Z}}(p), {{.ZMin}}) && {{.N.Le}}({{.N.Get3Z}}(p), {{.ZMax}});
				if (!insideZ) {
					if ({{.N.Gt}}(sdf2d, {{.N.Zero}})) {
						return {{.N.Sub}}({{.N.Zero}}, zDist);
					} else {
						return {{.N.Sub}}({{.N.Zero}}, {{.N.Sqrt}}({{.N.Add}}({{.N.Mul}}(zDist, zDist), {{.N.Mul}}(sdf2d, sdf2d))));
					}
				}
				if ({{.N.Gt}}(sdf2d, {{.N.Zero}})) {
					return {{.N.Min}}(sdf2d, zDist);
				} else {
					return sdf2d;
				}
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "Inner", k.EntrypointName, "ZMin", n.Literal(z0), "ZMax", n.Literal(z1))
	k.Kind = SDF3D
	k.EntrypointName = fnName
	return k
}

// RevolveSDF revolves a 2D SDF around the Z axis, where the x-axis becomes the
// radius axis and the y-axis becomes the z-axis. The left and right sides of
// the 2D profile are unioned, matching model3d.RevolveSDF.
func RevolveSDF(n Numerics, k ShapeKernel) ShapeKernel {
	if k.Kind != SDF2D {
		panic("expected 2D SDF kernel")
	}
	fnName := genFunctionID(&k.IDs, "revolve_sdf")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> {{.N.Dtype}} {
				let r = {{.N.Len2}}({{.N.Make2}}({{.N.Get3X}}(p), {{.N.Get3Y}}(p)));
				let z = {{.N.Get3Z}}(p);
				let dPos = {{.Inner}}({{.N.Make2}}(r, z));
				let dNeg = {{.Inner}}({{.N.Make2}}({{.N.Sub}}({{.N.Zero}}, r), z));
				return {{.N.Max}}(dPos, dNeg);
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "Inner", k.EntrypointName)
	k.Kind = SDF3D
	k.EntrypointName = fnName
	return k
}

// RevolveSolid revolves a 2D shape fully around the Z axis.
func RevolveSolid(n Numerics, k ShapeKernel) ShapeKernel {
	return RevolveSolidRange(n, k, 2*math.Pi, 0)
}

// RevolveSolidRange revolves a 2D shape around the Z axis with a start angle
// and total sweep in radians, matching model3d.RevolveSolidRange.
func RevolveSolidRange(n Numerics, k ShapeKernel, angleRad float64, startRad float64) ShapeKernel {
	switch k.Kind {
	case SDF2D:
		k = SDFToSolid(n, k)
	case Solid2D:
	default:
		panic("expected 2D solid or SDF kernel")
	}

	if math.Abs(angleRad) >= 2*math.Pi-1e-9 {
		fnName := genFunctionID(&k.IDs, "revolve_solid")
		AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
				let r = {{.N.Len2}}({{.N.Make2}}({{.N.Get3X}}(p), {{.N.Get3Y}}(p)));
				let z = {{.N.Get3Z}}(p);
				return {{.Inner}}({{.N.Make2}}(r, z)) ||
					{{.Inner}}({{.N.Make2}}({{.N.Sub}}({{.N.Zero}}, r), z));
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "Inner", k.EntrypointName)
		k.Kind = Solid3D
		k.EntrypointName = fnName
		return k
	}

	normalizeName := genFunctionID(&k.IDs, "normalize_angle")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(a: {{.N.Dtype}}) -> {{.N.Dtype}} {
				var result = a;
				for (var i = 0; i < 32; i++) {
					if (!{{.N.Lt}}(result, {{.N.Zero}})) {
						break;
					}
					result = {{.N.Add}}(result, {{.TwoPi}});
				}
				for (var i = 0; i < 32; i++) {
					if (!{{.N.Ge}}(result, {{.TwoPi}})) {
						break;
					}
					result = {{.N.Sub}}(result, {{.TwoPi}});
				}
				return result;
			}
		`, "N", n.Symbols, "Entrypoint", normalizeName, "TwoPi", n.Literal(2*math.Pi))

	fnName := genFunctionID(&k.IDs, "revolve_solid_range")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
				let x = {{.N.Get3X}}(p);
				let y = {{.N.Get3Y}}(p);
				let z = {{.N.Get3Z}}(p);
				let r = {{.N.Len2}}({{.N.Make2}}(x, y));
				let angle = {{.Angle}};
				let start = {{.Normalize}}({{.Start}});

				let theta = {{.N.Atan2}}(y, x);
				if ({{.N.Ge}}({{.Angle}}, {{.N.Zero}})) {
					let delta = {{.Normalize}}({{.N.Sub}}(theta, start));
					if ({{.N.Gt}}(delta, {{.N.Add}}({{.Angle}}, {{.Epsilon}}))) {
						return false;
					}
				} else {
					let delta = {{.Normalize}}({{.N.Sub}}(start, theta));
					if ({{.N.Gt}}(delta, {{.N.Add}}({{.N.Sub}}({{.N.Zero}}, {{.Angle}}), {{.Epsilon}}))) {
						return false;
					}
				}

				return {{.Inner}}({{.N.Make2}}(r, z)) ||
					{{.Inner}}({{.N.Make2}}({{.N.Sub}}({{.N.Zero}}, r), z));
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"Angle", n.Literal(angleRad),
		"Epsilon", n.Literal(1e-9),
		"Normalize", normalizeName,
		"Start", n.Literal(startRad),
		"Inner", k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

func linearExtrudeZBounds(height float64, center bool) (float64, float64) {
	z0 := 0.0
	z1 := height
	if center {
		z0 = -height / 2
		z1 = height / 2
	}
	return z0, z1
}

// InsetExtrude turns a 2D SDF into a 3D solid with optional top and bottom
// chamfer or fillet insets/outsets.
func InsetExtrude(
	n Numerics,
	k ShapeKernel,
	height float64,
	center bool,
	bottom, top float64,
	bottomFn, topFn InsetFunction,
) ShapeKernel {
	if k.Kind != SDF2D {
		panic("expected 2D SDF kernel")
	}
	if height < 0 {
		height = -height
	}

	z0, z1 := linearExtrudeZBounds(height, center)
	bottomInsetName := genFunctionID(&k.IDs, "inset_extrude_bottom")
	topInsetName := genFunctionID(&k.IDs, "inset_extrude_top")
	k.Code += "\n" + insetExtrudeSideCode(n, bottomInsetName, z0, z1, bottom, true, bottomFn)
	k.Code += "\n" + insetExtrudeSideCode(n, topInsetName, z0, z1, top, false, topFn)

	fnName := genFunctionID(&k.IDs, "inset_extrude")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
				let z = {{.N.Get3Z}}(p);
				if ({{.N.Lt}}(z, {{.ZMin}}) || {{.N.Gt}}(z, {{.ZMax}})) {
					return false;
				}
				let inset = {{.N.Add}}({{.BottomInset}}(z), {{.TopInset}}(z));
				let p2d = {{.N.Make2}}({{.N.Get3X}}(p), {{.N.Get3Y}}(p));
				return {{.N.Gt}}({{.Inner}}(p2d), inset);
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"ZMin", n.Literal(z0),
		"ZMax", n.Literal(z1),
		"BottomInset", bottomInsetName,
		"TopInset", topInsetName,
		"Inner", k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

func insetExtrudeSideCode(n Numerics, fnName string, z0, z1, radius float64, bottom bool, kind InsetFunction) string {
	r := math.Abs(radius)
	outwards := radius < 0
	distExpr := WGSL("{{.N.Sub}}(z, {{.ZMin}})", "N", n.Symbols, "ZMin", n.Literal(z0))
	if !bottom {
		distExpr = WGSL("{{.N.Sub}}({{.ZMax}}, z)", "N", n.Symbols, "ZMax", n.Literal(z1))
	}

	var body string
	switch kind {
	case InsetExtrudeChamfer:
		if outwards {
			body = WGSL("return {{.N.Mul}}({{.Radius}}, {{.N.Sub}}(frac, {{.N.One}}));", "N", n.Symbols, "Radius", n.Literal(r))
		} else {
			body = WGSL("return {{.N.Mul}}({{.Radius}}, {{.N.Sub}}({{.N.One}}, frac));", "N", n.Symbols, "Radius", n.Literal(r))
		}
	case InsetExtrudeFillet:
		if outwards {
			body = WGSL("return {{.N.Mul}}({{.Radius}}, {{.N.Sub}}({{.N.Sqrt}}({{.N.Max}}({{.N.Zero}}, {{.N.Sub}}({{.N.One}}, {{.N.Mul}}(x, x)))), {{.N.One}}));", "N", n.Symbols, "Radius", n.Literal(r))
		} else {
			body = WGSL("return {{.N.Mul}}({{.Radius}}, {{.N.Sub}}({{.N.One}}, {{.N.Sqrt}}({{.N.Max}}({{.N.Zero}}, {{.N.Sub}}({{.N.One}}, {{.N.Mul}}(x, x))))));", "N", n.Symbols, "Radius", n.Literal(r))
		}
	default:
		panic(`inset extrude function must be "chamfer" or "fillet"`)
	}

	return WGSL(`
			fn {{.Entrypoint}}(z: {{.N.Dtype}}) -> {{.N.Dtype}} {
				if ({{.N.Le}}({{.Radius}}, {{.N.Zero}})) {
					return {{.N.Zero}};
				}
				let dist = {{.DistExpr}};
				if ({{.N.Ge}}(dist, {{.Radius}})) {
					return {{.N.Zero}};
				}
				let frac = {{.N.Clamp}}({{.N.Div}}(dist, {{.Radius}}), {{.N.Zero}}, {{.N.One}});
				let x = {{.N.Sub}}(frac, {{.N.One}});
				{{.Body}}
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "Radius", n.Literal(r), "DistExpr", distExpr, "Body", body)
}
