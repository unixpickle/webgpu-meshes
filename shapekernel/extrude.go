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
func LinearExtrudeSolid(n Numerics, k ShapeKernel, height float32, center bool, twist float32, scale Vec2) ShapeKernel {
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
		"ZMin", n.Literal(float64(z0)),
		"ZMax", n.Literal(float64(z1)),
		"Height", n.Literal(float64(height)),
		"ScaleX", n.Literal(scale[0]),
		"ScaleY", n.Literal(scale[1]),
		"Twist", n.Literal(float64(twist)),
		"Inner", k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

// LinearExtrudeSDF turns a 2D SDF into a 3D SDF by extruding it along the Z
// axis with a height and optional centering.
func LinearExtrudeSDF(n Numerics, k ShapeKernel, height float32, center bool) ShapeKernel {
	if k.Kind != SDF2D {
		panic("expected 2D SDF kernel")
	}
	if height < 0 {
		height = -height
	}
	z0, z1 := linearExtrudeZBounds(height, center)
	fnName := genFunctionID(&k.IDs, "linear_extrude_sdf")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.N.Dtype3}}) -> {{.N.Dtype}} {
				let p = {{.N.AsFloat3}}(pRaw);
				let p2d = {{.N.Make2}}({{.N.FromFloat}}(p.x), {{.N.FromFloat}}(p.y));
				let sdf2d = {{.N.AsFloat}}({{.Inner}}(p2d));
				let zDist = min(abs(p.z - {{.ZMin}}), abs(p.z - {{.ZMax}}));
				let insideZ = p.z >= {{.ZMin}} && p.z <= {{.ZMax}};
				if (!insideZ) {
					if (sdf2d > 0.0) {
						return {{.N.FromFloat}}(-zDist);
					} else {
						return {{.N.FromFloat}}(-sqrt(zDist * zDist + sdf2d * sdf2d));
					}
				}
				if (sdf2d > 0.0) {
					return {{.N.FromFloat}}(min(sdf2d, zDist));
				} else {
					return {{.N.FromFloat}}(sdf2d);
				}
			}
		`, "N", n.Symbols, "Entrypoint", fnName, "Inner", k.EntrypointName, "ZMin", z0, "ZMax", z1)
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
			fn {{.Entrypoint}}(pRaw: {{.N.Dtype3}}) -> {{.N.Dtype}} {
				let p = {{.N.AsFloat3}}(pRaw);
				let r = length(p.xy);
				let dPos = {{.N.AsFloat}}({{.Inner}}({{.N.Make2}}({{.N.FromFloat}}(r), {{.N.FromFloat}}(p.z))));
				let dNeg = {{.N.AsFloat}}({{.Inner}}({{.N.Make2}}({{.N.FromFloat}}(-r), {{.N.FromFloat}}(p.z))));
				return {{.N.FromFloat}}(max(dPos, dNeg));
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
func RevolveSolidRange(n Numerics, k ShapeKernel, angleRad float32, startRad float32) ShapeKernel {
	switch k.Kind {
	case SDF2D:
		k = SDFToSolid(n, k)
	case Solid2D:
	default:
		panic("expected 2D solid or SDF kernel")
	}

	normalizeName := genFunctionID(&k.IDs, "normalize_angle")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(a: f32) -> f32 {
				let twoPi = 6.283185307179586;
				var result = a - floor(a / twoPi) * twoPi;
				if (result < 0.0) {
					result += twoPi;
				}
				return result;
			}
		`, "Entrypoint", normalizeName)

	fnName := genFunctionID(&k.IDs, "revolve_solid_range")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.N.Dtype3}}) -> bool {
				let p = {{.N.AsFloat3}}(pRaw);
				let r = length(p.xy);
				let angle = {{.Angle}};
				let start = {{.Normalize}}({{.Start}});
				let full = abs(angle) >= 6.283185307179586 - 1e-9;

				if (!full) {
					let theta = atan2(p.y, p.x);
					if (angle >= 0.0) {
						let delta = {{.Normalize}}(theta - start);
						if (delta > angle + 1e-9) {
							return false;
						}
					} else {
						let delta = {{.Normalize}}(start - theta);
						if (delta > -angle + 1e-9) {
							return false;
						}
					}
				}

				return {{.Inner}}({{.N.Make2}}({{.N.FromFloat}}(r), {{.N.FromFloat}}(p.z))) ||
					{{.Inner}}({{.N.Make2}}({{.N.FromFloat}}(-r), {{.N.FromFloat}}(p.z)));
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"Angle", angleRad,
		"Normalize", normalizeName,
		"Start", startRad,
		"Inner", k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

func linearExtrudeZBounds(height float32, center bool) (float32, float32) {
	z0 := float32(0.0)
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
	height float32,
	center bool,
	bottom, top float32,
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
	k.Code += "\n" + insetExtrudeSideCode(bottomInsetName, z0, z1, bottom, true, bottomFn)
	k.Code += "\n" + insetExtrudeSideCode(topInsetName, z0, z1, top, false, topFn)

	fnName := genFunctionID(&k.IDs, "inset_extrude")
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(pRaw: {{.N.Dtype3}}) -> bool {
				let p = {{.N.AsFloat3}}(pRaw);
				if (p.z < {{.ZMin}} || p.z > {{.ZMax}}) {
					return false;
				}
				let inset = {{.BottomInset}}(p.z) + {{.TopInset}}(p.z);
				let p2d = {{.N.Make2}}({{.N.FromFloat}}(p.x), {{.N.FromFloat}}(p.y));
				return {{.N.AsFloat}}({{.Inner}}(p2d)) > inset;
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"ZMin", z0,
		"ZMax", z1,
		"BottomInset", bottomInsetName,
		"TopInset", topInsetName,
		"Inner", k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

func insetExtrudeSideCode(fnName string, z0, z1, radius float32, bottom bool, kind InsetFunction) string {
	r := float32(math.Abs(float64(radius)))
	outwards := radius < 0
	distExpr := WGSL("z - {{.ZMin}}", "ZMin", z0)
	if !bottom {
		distExpr = WGSL("{{.ZMax}} - z", "ZMax", z1)
	}

	var body string
	switch kind {
	case InsetExtrudeChamfer:
		if outwards {
			body = WGSL("return {{.Radius}} * (frac - 1.0);", "Radius", r)
		} else {
			body = WGSL("return {{.Radius}} * (1.0 - frac);", "Radius", r)
		}
	case InsetExtrudeFillet:
		if outwards {
			body = WGSL("return {{.Radius}} * (sqrt(max(0.0, 1.0 - x*x)) - 1.0);", "Radius", r)
		} else {
			body = WGSL("return {{.Radius}} * (1.0 - sqrt(max(0.0, 1.0 - x*x)));", "Radius", r)
		}
	default:
		panic(`inset extrude function must be "chamfer" or "fillet"`)
	}

	return WGSL(`
			fn {{.Entrypoint}}(z: f32) -> f32 {
				if ({{.Radius}} <= 0.0) {
					return 0.0;
				}
				let dist = {{.DistExpr}};
				if (dist >= {{.Radius}}) {
					return 0.0;
				}
				let frac = clamp(dist / {{.Radius}}, 0.0, 1.0);
				let x = frac - 1.0;
				{{.Body}}
			}
		`, "Entrypoint", fnName, "Radius", r, "DistExpr", distExpr, "Body", body)
}
