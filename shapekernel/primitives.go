package shapekernel

import (
	"fmt"
)

func flattenSegment3s(lines []Segment3) []float32 {
	result := make([]float32, 0, len(lines)*6)
	for _, line := range lines {
		result = append(result, line[0][0], line[0][1], line[0][2])
		result = append(result, line[1][0], line[1][1], line[1][2])
	}
	return result
}

func Empty(kind ShapeKind) ShapeKernel {
	switch kind {
	case Solid2D, Solid3D:
		return emptyKernel(kind, "empty_solid", "false")
	case SDF2D, SDF3D:
		return emptyKernel(kind, "empty_sdf", "-1.0 / (p.x - p.x)")
	default:
		panic("expected solid or SDF kind")
	}
}

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

func emptyKernel(kind ShapeKind, name, returnExpr string) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, name)
	return ShapeKernel{
		Kind: kind,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: %s) -> %s {
					return %s;
				}
			`),
			entrypointName,
			kind.ArgType(),
			kind.ReturnType(),
			returnExpr,
		),
		EntrypointName: entrypointName,
	}
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

func Teardrop2DSolid(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "teardrop2d_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> bool {
					if (length(p) <= %f) {
						return true;
					}
					return p.y >= %f && p.y + abs(p.x) <= %f;
				}
			`),
			entrypointName,
			r,
			r/1.4142135623730951,
			r*1.4142135623730951,
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

// LineJoinSolid creates a solid containing all points within Euclidean
// distance r of any segment, like toolbox3d.LineJoin.
func LineJoinSolid(r float32, lines ...Segment3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "line_join_solid")
	bufName := genBufferID(&ids, "segments")
	segmentDistanceName := genFunctionID(&ids, "segment_distance3d")
	lineData := flattenSegment3s(lines)

	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Buffers: []Buffer{
			Float32Buffer(bufName, func() []float32 {
				return lineData
			}),
		},
		Code: Dedent(fmt.Sprintf(`
			fn %s(p: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
				let v = p2 - p1;
				let vNormSq = dot(v, v);
				var t = 0.0;
				if (vNormSq > 0.0) {
					t = clamp(dot(p - p1, v) / vNormSq, 0.0, 1.0);
				}
				let closest = p1 + t * v;
				return distance(p, closest);
			}

				fn %s(p: vec3<f32>) -> bool {
					let numSegs = %du;
					for (var i = 0u; i < numSegs; i++) {
						let p1 = vec3<f32>(%s[i*6], %s[i*6+1], %s[i*6+2]);
						let p2 = vec3<f32>(%s[i*6+3], %s[i*6+4], %s[i*6+5]);
						if (%s(p, p1, p2) <= %f) {
							return true;
						}
					}
					return false;
				}
		`, segmentDistanceName, entrypointName, len(lines),
			bufName, bufName, bufName, bufName, bufName, bufName,
			segmentDistanceName, r)),
		EntrypointName: entrypointName,
	}
}

// L1LineJoinSolid creates a solid containing all points within L1 distance r
// of any segment, like toolbox3d.L1LineJoin.
func L1LineJoinSolid(r float32, lines ...Segment3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "l1_line_join_solid")
	bufName := genBufferID(&ids, "segments")
	l1DistanceName := genFunctionID(&ids, "l1_distance3d")
	segmentL1DistanceName := genFunctionID(&ids, "segment_l1_distance3d")
	lineData := flattenSegment3s(lines)

	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Buffers: []Buffer{
			Float32Buffer(bufName, func() []float32 {
				return lineData
			}),
		},
		Code: Dedent(fmt.Sprintf(`
			fn %s(p1: vec3<f32>, p2: vec3<f32>) -> f32 {
				let delta = abs(p1 - p2);
				return delta.x + delta.y + delta.z;
			}

			fn %s(p: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
				let v = p2 - p1;
				var best = min(%s(p, p1), %s(p, p2));

				if (abs(v.x) > 1e-12) {
					let t = (p.x - p1.x) / v.x;
					if (t > 0.0 && t < 1.0) {
						best = min(best, %s(p, p1 + v * t));
					}
				}
				if (abs(v.y) > 1e-12) {
					let t = (p.y - p1.y) / v.y;
					if (t > 0.0 && t < 1.0) {
						best = min(best, %s(p, p1 + v * t));
					}
				}
				if (abs(v.z) > 1e-12) {
					let t = (p.z - p1.z) / v.z;
					if (t > 0.0 && t < 1.0) {
						best = min(best, %s(p, p1 + v * t));
					}
				}

				return best;
			}

				fn %s(p: vec3<f32>) -> bool {
					let numSegs = %du;
					for (var i = 0u; i < numSegs; i++) {
						let p1 = vec3<f32>(%s[i*6], %s[i*6+1], %s[i*6+2]);
						let p2 = vec3<f32>(%s[i*6+3], %s[i*6+4], %s[i*6+5]);
						let axisVec = p1 - p2;
					let axisLen = length(axisVec);
					if (axisLen <= 0.0) {
						if (%s(p, p1) < %f) {
							return true;
						}
						continue;
					}

					let axis = axisVec / axisLen;
					let axial = dot(p - p2, axis);
					if (axial >= 0.0 && axial <= axisLen && %s(p, p1, p2) < %f) {
						return true;
					}
					if (%s(p, p1) < %f || %s(p, p2) < %f) {
						return true;
					}
					}
					return false;
				}
		`, l1DistanceName, segmentL1DistanceName, l1DistanceName, l1DistanceName, l1DistanceName, l1DistanceName, l1DistanceName, entrypointName, len(lines),
			bufName, bufName, bufName, bufName, bufName, bufName,
			l1DistanceName, r, segmentL1DistanceName, r, l1DistanceName, r, l1DistanceName, r)),
		EntrypointName: entrypointName,
	}
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
