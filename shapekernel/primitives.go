package shapekernel

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
	AppendWGSL(&k, `
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> bool {
				return {{.Inner}}(p) >= 0.0;
			}
		`, "Entrypoint", fnName, "ArgType", argType, "Inner", k.EntrypointName)
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
		Code: WGSL(`
				fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
					return {{.ReturnExpr}};
				}
			`, "Entrypoint", entrypointName, "ArgType", kind.ArgType(), "ReturnType", kind.ReturnType(), "ReturnExpr", returnExpr),
		EntrypointName: entrypointName,
	}
}

func CircleSolid(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "circle_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec2<f32>) -> bool {
					let center = vec2<f32>(0.0, 0.0);
					return distance(p, center) <= {{.Radius}};
				}
			`, "Entrypoint", entrypointName, "Radius", r),
		EntrypointName: entrypointName,
	}
}

func Teardrop2DSolid(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "teardrop2d_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec2<f32>) -> bool {
					if (length(p) <= {{.Radius}}) {
						return true;
					}
					return p.y >= {{.YMin}} && p.y + abs(p.x) <= {{.DiagBound}};
				}
			`, "Entrypoint", entrypointName, "Radius", r, "YMin", r/1.4142135623730951, "DiagBound", r*1.4142135623730951),
		EntrypointName: entrypointName,
	}
}

func CircleSDF(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "circle_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec2<f32>) -> f32 {
					let center = vec2<f32>(0.0, 0.0);
					return {{.Radius}} - distance(p, center);
				}
			`, "Entrypoint", entrypointName, "Radius", r),
		EntrypointName: entrypointName,
	}
}

func Rect2DSolid(sideLengths Vec2) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect2d_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec2<f32>) -> bool {
					let halfSize = {{.SideLengths}} / 2.0;
					return all(abs(p) <= halfSize);
				}
			`, "Entrypoint", entrypointName, "SideLengths", sideLengths.WebGPUVec()),
		EntrypointName: entrypointName,
	}
}

func Rect2DSDF(sideLengths Vec2) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect2d_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec2<f32>) -> f32 {
					let halfSize = {{.SideLengths}} / 2.0;
					let q = abs(p) - halfSize;
					let outside = length(max(q, vec2<f32>(0.0, 0.0)));
					return -outside - min(max(q.x, q.y), 0.0);
				}
			`, "Entrypoint", entrypointName, "SideLengths", sideLengths.WebGPUVec()),
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
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec2<f32>) -> f32 {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = b - a;
					let pa = p - a;
					let lenSq = dot(ba, ba);
					if (lenSq <= 0.0) {
						return {{.Radius}} - distance(p, a);
					}
					let h = clamp(dot(pa, ba) / lenSq, 0.0, 1.0);
					return {{.Radius}} - length(pa - ba * h);
				}
			`, "Entrypoint", entrypointName, "P1", p1.WebGPUVec(), "P2", p2.WebGPUVec(), "Radius", radius),
		EntrypointName: entrypointName,
	}
}

func SphereSolid(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "sphere_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let center = vec3<f32>(0.0, 0.0, 0.0);
					return distance(p, center) <= {{.Radius}};
				}
			`, "Entrypoint", entrypointName, "Radius", r),
		EntrypointName: entrypointName,
	}
}

func Rect3DSolid(sideLengths Vec3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect3d_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let halfSize = {{.SideLengths}} / 2.0;
					return all(abs(p) <= halfSize);
				}
			`, "Entrypoint", entrypointName, "SideLengths", sideLengths.WebGPUVec()),
		EntrypointName: entrypointName,
	}
}

func Rect3DSDF(sideLengths Vec3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect3d_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> f32 {
					let halfSize = {{.SideLengths}} / 2.0;
					let q = abs(p) - halfSize;
					let outside = length(max(q, vec3<f32>(0.0, 0.0, 0.0)));
					return -outside - min(max(max(q.x, q.y), q.z), 0.0);
				}
			`, "Entrypoint", entrypointName, "SideLengths", sideLengths.WebGPUVec()),
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
		Code: WGSL(`
			fn {{.SegmentDistance}}(p: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
				let v = p2 - p1;
				let vNormSq = dot(v, v);
				var t = 0.0;
				if (vNormSq > 0.0) {
					t = clamp(dot(p - p1, v) / vNormSq, 0.0, 1.0);
				}
				let closest = p1 + t * v;
				return distance(p, closest);
			}

				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let numSegs = {{.NumSegs}}u;
					for (var i = 0u; i < numSegs; i++) {
						let p1 = vec3<f32>({{.Segments}}[i*6], {{.Segments}}[i*6+1], {{.Segments}}[i*6+2]);
						let p2 = vec3<f32>({{.Segments}}[i*6+3], {{.Segments}}[i*6+4], {{.Segments}}[i*6+5]);
						if ({{.SegmentDistance}}(p, p1, p2) <= {{.Radius}}) {
							return true;
						}
					}
					return false;
				}
		`, "SegmentDistance", segmentDistanceName, "Entrypoint", entrypointName, "NumSegs", len(lines), "Segments", bufName, "Radius", r),
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
		Code: WGSL(`
			fn {{.L1Distance}}(p1: vec3<f32>, p2: vec3<f32>) -> f32 {
				let delta = abs(p1 - p2);
				return delta.x + delta.y + delta.z;
			}

			fn {{.SegmentL1Distance}}(p: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
				let v = p2 - p1;
				var best = min({{.L1Distance}}(p, p1), {{.L1Distance}}(p, p2));

				if (abs(v.x) > 1e-12) {
					let t = (p.x - p1.x) / v.x;
					if (t > 0.0 && t < 1.0) {
						best = min(best, {{.L1Distance}}(p, p1 + v * t));
					}
				}
				if (abs(v.y) > 1e-12) {
					let t = (p.y - p1.y) / v.y;
					if (t > 0.0 && t < 1.0) {
						best = min(best, {{.L1Distance}}(p, p1 + v * t));
					}
				}
				if (abs(v.z) > 1e-12) {
					let t = (p.z - p1.z) / v.z;
					if (t > 0.0 && t < 1.0) {
						best = min(best, {{.L1Distance}}(p, p1 + v * t));
					}
				}

				return best;
			}

				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let numSegs = {{.NumSegs}}u;
					for (var i = 0u; i < numSegs; i++) {
						let p1 = vec3<f32>({{.Segments}}[i*6], {{.Segments}}[i*6+1], {{.Segments}}[i*6+2]);
						let p2 = vec3<f32>({{.Segments}}[i*6+3], {{.Segments}}[i*6+4], {{.Segments}}[i*6+5]);
						let axisVec = p1 - p2;
					let axisLen = length(axisVec);
					if (axisLen <= 0.0) {
						if ({{.L1Distance}}(p, p1) < {{.Radius}}) {
							return true;
						}
						continue;
					}

					let axis = axisVec / axisLen;
					let axial = dot(p - p2, axis);
					if (axial >= 0.0 && axial <= axisLen && {{.SegmentL1Distance}}(p, p1, p2) < {{.Radius}}) {
						return true;
					}
					if ({{.L1Distance}}(p, p1) < {{.Radius}} || {{.L1Distance}}(p, p2) < {{.Radius}}) {
						return true;
					}
					}
					return false;
				}
		`,
			"L1Distance", l1DistanceName,
			"SegmentL1Distance", segmentL1DistanceName,
			"Entrypoint", entrypointName,
			"NumSegs", len(lines),
			"Segments", bufName,
			"Radius", r,
		),
		EntrypointName: entrypointName,
	}
}

func Capsule3DSDF(p1, p2 Vec3, radius float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "capsule3d_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> f32 {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = b - a;
					let pa = p - a;
					let lenSq = dot(ba, ba);
					if (lenSq <= 0.0) {
						return {{.Radius}} - distance(p, a);
					}
					let h = clamp(dot(pa, ba) / lenSq, 0.0, 1.0);
					return {{.Radius}} - length(pa - ba * h);
				}
			`, "Entrypoint", entrypointName, "P1", p1.WebGPUVec(), "P2", p2.WebGPUVec(), "Radius", radius),
		EntrypointName: entrypointName,
	}
}

func CylinderSolid(p1, p2 Vec3, radius float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cylinder_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = b - a;
					let h = length(ba);
					if (h <= 0.0) {
						return distance(p, a) <= {{.Radius}};
					}
					let axis = ba / h;
					let axial = dot(p - a, axis);
					if (axial < 0.0 || axial > h) {
						return false;
					}
					let projection = a + axis * axial;
					return distance(p, projection) <= {{.Radius}};
				}
			`, "Entrypoint", entrypointName, "P1", p1.WebGPUVec(), "P2", p2.WebGPUVec(), "Radius", radius),
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
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let tip = {{.Tip}};
					let base = {{.Base}};
					let axisVec = tip - base;
					let h = length(axisVec);
					if (h <= 0.0) {
						return distance(p, tip) <= {{.Radius}};
					}
					let axis = axisVec / h;
					let axial = dot(p - base, axis);
					let radiusFrac = 1.0 - axial / h;
					if (radiusFrac < 0.0 || radiusFrac > 1.0) {
						return false;
					}
					let projection = base + axis * axial;
					return distance(p, projection) <= {{.Radius}} * radiusFrac;
				}
			`, "Entrypoint", entrypointName, "Tip", tip.WebGPUVec(), "Base", base.WebGPUVec(), "Radius", radius),
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
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> bool {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = b - a;
					let h = length(ba);
					let maxRadius = max({{.Radius1}}, {{.Radius2}});
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
					let radius = {{.Radius1}} * radiusFrac + {{.Radius2}} * (1.0 - radiusFrac);
					return distance(p, projection) <= radius;
				}
			`, "Entrypoint", entrypointName, "P1", p1.WebGPUVec(), "P2", p2.WebGPUVec(), "Radius1", r1, "Radius2", r2),
		EntrypointName: entrypointName,
	}
}

func ConeSliceSDF(p1, p2 Vec3, r1, r2 float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_slice_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> f32 {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = b - a;
					let h = length(ba);
					let maxRadius = max({{.Radius1}}, {{.Radius2}});
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
						let radius = {{.Radius1}} + ({{.Radius2}} - {{.Radius1}}) * axialFrac;
						inside = radial <= radius;
					}

					let q = vec2<f32>(axial, radial);
					let sideA = vec2<f32>(0.0, {{.Radius1}});
					let sideB = vec2<f32>(h, {{.Radius2}});
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
					if (q.y < {{.Radius1}}) {
						cap1Dist = abs(cap1Axial);
					} else {
						cap1Dist = length(vec2<f32>(cap1Axial, q.y - {{.Radius1}}));
					}

					let cap2Axial = q.x - h;
					var cap2Dist = length(vec2<f32>(cap2Axial, q.y));
					if (q.y < {{.Radius2}}) {
						cap2Dist = abs(cap2Axial);
					} else {
						cap2Dist = length(vec2<f32>(cap2Axial, q.y - {{.Radius2}}));
					}

					let unsignedDist = min(sideDist, min(cap1Dist, cap2Dist));
					if (inside) {
						return unsignedDist;
					} else {
						return -unsignedDist;
					}
				}
			`, "Entrypoint", entrypointName, "P1", p1.WebGPUVec(), "P2", p2.WebGPUVec(), "Radius1", r1, "Radius2", r2),
		EntrypointName: entrypointName,
	}
}

func SphereSDF(r float32) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "sphere_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(`
				fn {{.Entrypoint}}(p: vec3<f32>) -> f32 {
					let center = vec3<f32>(0.0, 0.0, 0.0);
					return {{.Radius}} - distance(p, center);
				}
			`, "Entrypoint", entrypointName, "Radius", r),
		EntrypointName: entrypointName,
	}
}
