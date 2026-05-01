package shapekernel

import (
	"fmt"
)

func SDFToSolid(k ShapeKernel) ShapeKernel {
	argType := k.Kind.ArgType()
	solidKind := Solid2D
	switch k.Kind {
	case SDF2D:
		solidKind = Solid2D
	case SDF3D:
		solidKind = Solid3D
	default:
		panic("expected SDF kernel")
	}
	fnName := genFunctionID(&k.IDs, "sdf_to_solid")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> bool {
				return %s(p) >= 0.0;
			}
		`),
		fnName,
		argType,
		k.EntrypointName,
	)
	k.Kind = solidKind
	k.EntrypointName = fnName
	return k
}

func CircleSolid(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "circle_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> bool {
					let center = vec2<f32>(0.0, 0.0);
					return distance(p, center) <= %f;
				}
			`),
			entrypointName,
			r,
		),
		EntrypointName: entrypointName,
	}
}

func CircleSDF(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "circle_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> f32 {
					let center = vec2<f32>(0.0, 0.0);
					return %f - distance(p, center);
				}
			`),
			entrypointName,
			r,
		),
		EntrypointName: entrypointName,
	}
}

func Rect2DSolid(sideLengths Vec2) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect2d_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> bool {
					let halfSize = %s / 2.0;
					return all(abs(p) <= halfSize);
				}
			`),
			entrypointName,
			sideLengths.WebGPUVec(),
		),
		EntrypointName: entrypointName,
	}
}

func Rect2DSDF(sideLengths Vec2) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect2d_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> f32 {
					let halfSize = %s / 2.0;
					let q = abs(p) - halfSize;
					let outside = length(max(q, vec2<f32>(0.0, 0.0)));
					return -outside - min(max(q.x, q.y), 0.0);
				}
			`),
			entrypointName,
			sideLengths.WebGPUVec(),
		),
		EntrypointName: entrypointName,
	}
}

func Capsule2DSolid(p1, p2 Vec2, radius float32) ShapeKernel {
	return SDFToSolid(Capsule2DSDF(p1, p2, radius))
}

func Capsule2DSDF(p1, p2 Vec2, radius float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "capsule2d_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> f32 {
					let a = %s;
					let b = %s;
					let ba = b - a;
					let pa = p - a;
					let lenSq = dot(ba, ba);
					if (lenSq <= 0.0) {
						return %f - distance(p, a);
					}
					let h = clamp(dot(pa, ba) / lenSq, 0.0, 1.0);
					return %f - length(pa - ba * h);
				}
			`),
			entrypointName,
			p1.WebGPUVec(),
			p2.WebGPUVec(),
			radius,
			radius,
		),
		EntrypointName: entrypointName,
	}
}

func SphereSolid(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "sphere_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> bool {
					let center = vec3<f32>(0.0, 0.0, 0.0);
					return distance(p, center) <= %f;
				}
			`),
			entrypointName,
			r,
		),
		EntrypointName: entrypointName,
	}
}

func Rect3DSolid(sideLengths Vec3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect3d_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> bool {
					let halfSize = %s / 2.0;
					return all(abs(p) <= halfSize);
				}
			`),
			entrypointName,
			sideLengths.WebGPUVec(),
		),
		EntrypointName: entrypointName,
	}
}

func Rect3DSDF(sideLengths Vec3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect3d_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> f32 {
					let halfSize = %s / 2.0;
					let q = abs(p) - halfSize;
					let outside = length(max(q, vec3<f32>(0.0, 0.0, 0.0)));
					return -outside - min(max(max(q.x, q.y), q.z), 0.0);
				}
			`),
			entrypointName,
			sideLengths.WebGPUVec(),
		),
		EntrypointName: entrypointName,
	}
}

func Capsule3DSolid(p1, p2 Vec3, radius float32) ShapeKernel {
	return SDFToSolid(Capsule3DSDF(p1, p2, radius))
}

func Capsule3DSDF(p1, p2 Vec3, radius float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "capsule3d_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> f32 {
					let a = %s;
					let b = %s;
					let ba = b - a;
					let pa = p - a;
					let lenSq = dot(ba, ba);
					if (lenSq <= 0.0) {
						return %f - distance(p, a);
					}
					let h = clamp(dot(pa, ba) / lenSq, 0.0, 1.0);
					return %f - length(pa - ba * h);
				}
			`),
			entrypointName,
			p1.WebGPUVec(),
			p2.WebGPUVec(),
			radius,
			radius,
		),
		EntrypointName: entrypointName,
	}
}

func CylinderSolid(p1, p2 Vec3, radius float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cylinder_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> bool {
					let a = %s;
					let b = %s;
					let ba = b - a;
					let h = length(ba);
					if (h <= 0.0) {
						return distance(p, a) <= %f;
					}
					let axis = ba / h;
					let axial = dot(p - a, axis);
					if (axial < 0.0 || axial > h) {
						return false;
					}
					let projection = a + axis * axial;
					return distance(p, projection) <= %f;
				}
			`),
			entrypointName,
			p1.WebGPUVec(),
			p2.WebGPUVec(),
			radius,
			radius,
		),
		EntrypointName: entrypointName,
	}
}

func CylinderSDF(p1, p2 Vec3, radius float32) ShapeKernel {
	return ConeSliceSDF(p1, p2, radius, radius)
}

func ConeSolid(tip, base Vec3, radius float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> bool {
					let tip = %s;
					let base = %s;
					let axisVec = tip - base;
					let h = length(axisVec);
					if (h <= 0.0) {
						return distance(p, tip) <= %f;
					}
					let axis = axisVec / h;
					let axial = dot(p - base, axis);
					let radiusFrac = 1.0 - axial / h;
					if (radiusFrac < 0.0 || radiusFrac > 1.0) {
						return false;
					}
					let projection = base + axis * axial;
					return distance(p, projection) <= %f * radiusFrac;
				}
			`),
			entrypointName,
			tip.WebGPUVec(),
			base.WebGPUVec(),
			radius,
			radius,
		),
		EntrypointName: entrypointName,
	}
}

func ConeSDF(tip, base Vec3, radius float32) ShapeKernel {
	return ConeSliceSDF(tip, base, 0.0, radius)
}

func ConeSliceSolid(p1, p2 Vec3, r1, r2 float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_slice_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> bool {
					let a = %s;
					let b = %s;
					let ba = b - a;
					let h = length(ba);
					let maxRadius = max(%f, %f);
					if (h <= 0.0) {
						return distance(p, a) <= maxRadius;
					}
					let axis = ba / h;
					let axial = dot(p - a, axis);
					let radiusFrac = 1.0 - axial / h;
					if (radiusFrac < 0.0 || radiusFrac > 1.0) {
						return false;
					}
					let projection = a + axis * axial;
					let radius = %f * radiusFrac + %f * (1.0 - radiusFrac);
					return distance(p, projection) <= radius;
				}
			`),
			entrypointName,
			p1.WebGPUVec(),
			p2.WebGPUVec(),
			r1,
			r2,
			r1,
			r2,
		),
		EntrypointName: entrypointName,
	}
}

func ConeSliceSDF(p1, p2 Vec3, r1, r2 float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_slice_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> f32 {
					let a = %s;
					let b = %s;
					let ba = b - a;
					let h = length(ba);
					let maxRadius = max(%f, %f);
					if (h <= 0.0) {
						return maxRadius - distance(p, a);
					}

					let axis = ba / h;
					let pa = p - a;
					let axial = dot(pa, axis);
					let radialVec = pa - axis * axial;
					let radial = length(radialVec);

					var inside = false;
					if (axial >= 0.0 && axial <= h) {
						let axialFrac = axial / h;
						let radius = %f + (%f - %f) * axialFrac;
						inside = radial <= radius;
					}

					let q = vec2<f32>(axial, radial);
					let sideA = vec2<f32>(0.0, %f);
					let sideB = vec2<f32>(h, %f);
					let sideBA = sideB - sideA;
					let sidePASeg = q - sideA;
					let sideLenSq = dot(sideBA, sideBA);
					var sideT = 0.0;
					if (sideLenSq > 0.0) {
						sideT = clamp(dot(sidePASeg, sideBA) / sideLenSq, 0.0, 1.0);
					}
					let sideDist = length(sidePASeg - sideBA * sideT);

					let cap1Axial = q.x;
					var cap1Dist = length(q);
					if (q.y < %f) {
						cap1Dist = abs(cap1Axial);
					} else {
						cap1Dist = length(vec2<f32>(cap1Axial, q.y - %f));
					}

					let cap2Axial = q.x - h;
					var cap2Dist = length(vec2<f32>(cap2Axial, q.y));
					if (q.y < %f) {
						cap2Dist = abs(cap2Axial);
					} else {
						cap2Dist = length(vec2<f32>(cap2Axial, q.y - %f));
					}

					let unsignedDist = min(sideDist, min(cap1Dist, cap2Dist));
					if (inside) {
						return unsignedDist;
					} else {
						return -unsignedDist;
					}
				}
			`),
			entrypointName,
			p1.WebGPUVec(),
			p2.WebGPUVec(),
			r1,
			r2,
			r1,
			r2,
			r1,
			r1,
			r2,
			r1,
			r1,
			r2,
			r2,
		),
		EntrypointName: entrypointName,
	}
}

func SphereSDF(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "sphere_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec3<f32>) -> f32 {
					let center = vec3<f32>(0.0, 0.0, 0.0);
					return %f - distance(p, center);
				}
			`),
			entrypointName,
			r,
		),
		EntrypointName: entrypointName,
	}
}
