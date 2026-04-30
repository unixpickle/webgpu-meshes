package shapekernel

import (
	"fmt"
	"math"
)

const (
	InsetExtrudeChamfer InsetFunction = "chamfer"
	InsetExtrudeFillet  InsetFunction = "fillet"
)

type InsetFunction string

// LinearExtrudeSolid extends a 2D shape along the Z axis, optionally centered,
// twisted, and scaled from bottom to top.
func LinearExtrudeSolid(k ShapeKernel, height float32, center bool, twist float32, scale Vec2) ShapeKernel {
	switch k.Kind {
	case SDF2D:
		k = solidFromSDF(k, "linear_extrude_source_solid")
	case Solid2D:
	default:
		panic("expected 2D solid or SDF kernel")
	}

	if height < 0 {
		height = -height
	}
	z0, z1 := linearExtrudeZBounds(height, center)
	fnName := genFunctionID(&k.IDs, "linear_extrude")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: vec3<f32>) -> bool {
				if (p.z < %f || p.z > %f) {
					return false;
				}

				var t = 0.0;
				if (%f > 0.0) {
					t = (p.z - %f) / %f;
				}

				let sx = 1.0 + t * (%f - 1.0);
				let sy = 1.0 + t * (%f - 1.0);
				if (sx == 0.0 || sy == 0.0) {
					return false;
				}

				let angle = %f * t;
				let cosA = cos(angle);
				let sinA = sin(angle);
				let rx = p.x * cosA - p.y * sinA;
				let ry = p.x * sinA + p.y * cosA;
				return %s(vec2<f32>(rx / sx, ry / sy));
			}
		`),
		fnName,
		z0,
		z1,
		height,
		z0,
		height,
		scale[0],
		scale[1],
		twist,
		k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

// LinearExtrudeSDF turns a 2D SDF into a 3D SDF by extruding it along the Z
// axis with a height and optional centering.
func LinearExtrudeSDF(k ShapeKernel, height float32, center bool) ShapeKernel {
	if k.Kind != SDF2D {
		panic("expected 2D SDF kernel")
	}
	if height < 0 {
		height = -height
	}
	z0, z1 := linearExtrudeZBounds(height, center)
	fnName := genFunctionID(&k.IDs, "linear_extrude_sdf")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: vec3<f32>) -> f32 {
				let sdf2d = %s(p.xy);
				let zDist = min(abs(p.z - %f), abs(p.z - %f));
				let insideZ = p.z >= %f && p.z <= %f;
				if (!insideZ) {
					if (sdf2d > 0.0) {
						return -zDist;
					} else {
						return -sqrt(zDist * zDist + sdf2d * sdf2d);
					}
				}
				if (sdf2d > 0.0) {
					return min(sdf2d, zDist);
				} else {
					return sdf2d;
				}
			}
		`),
		fnName,
		k.EntrypointName,
		z0,
		z1,
		z0,
		z1,
	)
	k.Kind = SDF3D
	k.EntrypointName = fnName
	return k
}

// RevolveSDF revolves a 2D SDF around the Z axis, where the x-axis becomes the
// radius axis and the y-axis becomes the z-axis. The left and right sides of
// the 2D profile are unioned, matching model3d.RevolveSDF.
func RevolveSDF(k ShapeKernel) ShapeKernel {
	if k.Kind != SDF2D {
		panic("expected 2D SDF kernel")
	}
	fnName := genFunctionID(&k.IDs, "revolve_sdf")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: vec3<f32>) -> f32 {
				let r = length(p.xy);
				let dPos = %s(vec2<f32>(r, p.z));
				let dNeg = %s(vec2<f32>(-r, p.z));
				return max(dPos, dNeg);
			}
		`),
		fnName,
		k.EntrypointName,
		k.EntrypointName,
	)
	k.Kind = SDF3D
	k.EntrypointName = fnName
	return k
}

// RevolveSolid revolves a 2D shape fully around the Z axis.
func RevolveSolid(k ShapeKernel) ShapeKernel {
	return RevolveSolidRange(k, 2*math.Pi, 0)
}

// RevolveSolidRange revolves a 2D shape around the Z axis with a start angle
// and total sweep in radians, matching model3d.RevolveSolidRange.
func RevolveSolidRange(k ShapeKernel, angleRad float32, startRad float32) ShapeKernel {
	switch k.Kind {
	case SDF2D:
		k = solidFromSDF(k, "revolve_solid_source_solid")
	case Solid2D:
	default:
		panic("expected 2D solid or SDF kernel")
	}

	normalizeName := genFunctionID(&k.IDs, "normalize_angle")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(a: f32) -> f32 {
				let twoPi = 6.283185307179586;
				var result = a - floor(a / twoPi) * twoPi;
				if (result < 0.0) {
					result += twoPi;
				}
				return result;
			}
		`),
		normalizeName,
	)

	fnName := genFunctionID(&k.IDs, "revolve_solid_range")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: vec3<f32>) -> bool {
				let r = length(p.xy);
				let angle = %f;
				let start = %s(%f);
				let full = abs(angle) >= 6.283185307179586 - 1e-9;

				if (!full) {
					let theta = atan2(p.y, p.x);
					if (angle >= 0.0) {
						let delta = %s(theta - start);
						if (delta > angle + 1e-9) {
							return false;
						}
					} else {
						let delta = %s(start - theta);
						if (delta > -angle + 1e-9) {
							return false;
						}
					}
				}

				return %s(vec2<f32>(r, p.z)) || %s(vec2<f32>(-r, p.z));
			}
		`),
		fnName,
		angleRad,
		normalizeName,
		startRad,
		normalizeName,
		normalizeName,
		k.EntrypointName,
		k.EntrypointName,
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
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: vec3<f32>) -> bool {
				if (p.z < %f || p.z > %f) {
					return false;
				}
				let inset = %s(p.z) + %s(p.z);
				return %s(p.xy) > inset;
			}
		`),
		fnName,
		z0,
		z1,
		bottomInsetName,
		topInsetName,
		k.EntrypointName,
	)
	k.Kind = Solid3D
	k.EntrypointName = fnName
	return k
}

func insetExtrudeSideCode(fnName string, z0, z1, radius float32, bottom bool, kind InsetFunction) string {
	r := float32(math.Abs(float64(radius)))
	outwards := radius < 0
	distExpr := fmt.Sprintf("z - %f", z0)
	if !bottom {
		distExpr = fmt.Sprintf("%f - z", z1)
	}

	var body string
	switch kind {
	case InsetExtrudeChamfer:
		if outwards {
			body = fmt.Sprintf("return %f * (frac - 1.0);", r)
		} else {
			body = fmt.Sprintf("return %f * (1.0 - frac);", r)
		}
	case InsetExtrudeFillet:
		if outwards {
			body = fmt.Sprintf("return %f * (sqrt(max(0.0, 1.0 - x*x)) - 1.0);", r)
		} else {
			body = fmt.Sprintf("return %f * (1.0 - sqrt(max(0.0, 1.0 - x*x)));", r)
		}
	default:
		panic(`inset extrude function must be "chamfer" or "fillet"`)
	}

	return fmt.Sprintf(
		Dedent(`
			fn %s(z: f32) -> f32 {
				if (%f <= 0.0) {
					return 0.0;
				}
				let dist = %s;
				if (dist >= %f) {
					return 0.0;
				}
				let frac = clamp(dist / %f, 0.0, 1.0);
				let x = frac - 1.0;
				%s
			}
		`),
		fnName,
		r,
		distExpr,
		r,
		r,
		body,
	)
}
