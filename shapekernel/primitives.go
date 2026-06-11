package shapekernel

func Empty(n Numerics, kind ShapeKind) ShapeKernel {
	switch kind {
	case Solid2D, Solid3D:
		return emptyKernel(n, kind, "empty_solid", "false")
	case SDF2D, SDF3D:
		return emptyKernel(n, kind, "empty_sdf", n.Infinity(-1))
	default:
		panic("expected solid or SDF kind")
	}
}

func SDFToSolid(n Numerics, k ShapeKernel) ShapeKernel {
	argType := k.Kind.ArgType(n)
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
	AppendWGSL(
		&k,
		`
			fn {{.Entrypoint}}(p: {{.ArgType}}) -> bool {
				return {{.N.Ge}}({{.Inner}}(p), {{.N.Zero}});
			}
		`,
		"N", n.Symbols,
		"Entrypoint", fnName,
		"ArgType", argType,
		"Inner", k.EntrypointName,
	)
	k.Kind = solidKind
	k.EntrypointName = fnName
	return k
}

func emptyKernel(n Numerics, kind ShapeKind, name, returnExpr string) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, name)
	return ShapeKernel{
		Kind: kind,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
					return {{.ReturnExpr}};
				}
			`,
			"Entrypoint", entrypointName,
			"ArgType", kind.ArgType(n),
			"ReturnType", kind.ReturnType(n),
			"ReturnExpr", returnExpr,
		),
		EntrypointName: entrypointName,
	}
}

func CircleSolid(n Numerics, r float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "circle_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype2}}) -> bool {
					let center = {{.N.Make2}}({{.N.Zero}}, {{.N.Zero}});
					return {{.N.Le}}({{.N.Dist2}}(p, center), {{.Radius}});
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"Radius", n.Literal(r),
		),
		EntrypointName: entrypointName,
	}
}

func Teardrop2DSolid(n Numerics, r float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "teardrop2d_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype2}}) -> bool {
					if ({{.N.Le}}({{.N.Len2}}(p), {{.Radius}})) {
						return true;
					}
					return {{.N.Ge}}({{.N.Get2Y}}(p), {{.YMin}}) && {{.N.Le}}({{.N.Add}}({{.N.Get2Y}}(p), {{.N.Abs}}({{.N.Get2X}}(p))), {{.DiagBound}});
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"Radius", n.Literal(r),
			"YMin", n.Literal(r/1.4142135623730951),
			"DiagBound", n.Literal(r*1.4142135623730951),
		),
		EntrypointName: entrypointName,
	}
}

func CircleSDF(n Numerics, r float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "circle_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype2}}) -> {{.N.Dtype}} {
					let center = {{.N.Make2}}({{.N.Zero}}, {{.N.Zero}});
					return {{.N.Sub}}({{.Radius}}, {{.N.Dist2}}(p, center));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"Radius", n.Literal(r),
		),
		EntrypointName: entrypointName,
	}
}

func Rect2DSolid(n Numerics, sideLengths Vec2) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect2d_solid")
	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype2}}) -> bool {
					let halfSize = {{.N.Scale2}}({{.SideLengths}}, {{.Half}});
					let absP = {{.N.Abs2}}(p);
					return {{.N.Le}}({{.N.Get2X}}(absP), {{.N.Get2X}}(halfSize)) && {{.N.Le}}({{.N.Get2Y}}(absP), {{.N.Get2Y}}(halfSize));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"SideLengths", sideLengths.WebGPUVec(n),
			"Half", n.Literal(0.5),
		),
		EntrypointName: entrypointName,
	}
}

func Rect2DSDF(n Numerics, sideLengths Vec2) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect2d_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype2}}) -> {{.N.Dtype}} {
					let halfSize = {{.N.Scale2}}({{.SideLengths}}, {{.Half}});
					let q = {{.N.Sub2}}({{.N.Abs2}}(p), halfSize);
					let outside = {{.N.Len2}}({{.N.Max2}}(q, {{.Zero2}}));
					let inside = {{.N.Min}}({{.N.Max}}({{.N.Get2X}}(q), {{.N.Get2Y}}(q)), {{.N.Zero}});
					return {{.N.Sub}}({{.N.Sub}}({{.N.Zero}}, outside), inside);
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"SideLengths", sideLengths.WebGPUVec(n),
			"Half", n.Literal(0.5),
			"Zero2", WGSL("{{.N.Make2}}({{.N.Zero}}, {{.N.Zero}})", "N", n.Symbols),
		),
		EntrypointName: entrypointName,
	}
}

func Capsule2DSolid(n Numerics, p1, p2 Vec2, radius float64) ShapeKernel {
	return SDFToSolid(n, Capsule2DSDF(n, p1, p2, radius))
}

func Capsule2DSDF(n Numerics, p1, p2 Vec2, radius float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "capsule2d_sdf")
	return ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype2}}) -> {{.N.Dtype}} {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = {{.N.Sub2}}(b, a);
					let pa = {{.N.Sub2}}(p, a);
					let lenSq = {{.N.Dot2}}(ba, ba);
					if ({{.N.Le}}(lenSq, {{.N.Zero}})) {
						return {{.N.Sub}}({{.Radius}}, {{.N.Dist2}}(p, a));
					}
					let h = {{.N.Clamp}}({{.N.Div}}({{.N.Dot2}}(pa, ba), lenSq), {{.N.Zero}}, {{.N.One}});
					return {{.N.Sub}}({{.Radius}}, {{.N.Len2}}({{.N.Sub2}}(pa, {{.N.Scale2}}(ba, h))));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"P1", p1.WebGPUVec(n),
			"P2", p2.WebGPUVec(n),
			"Radius", n.Literal(radius),
		),
		EntrypointName: entrypointName,
	}
}

func SphereSolid(n Numerics, r float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "sphere_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
					let center = {{.N.Make3}}({{.N.Zero}}, {{.N.Zero}}, {{.N.Zero}});
					return {{.N.Le}}({{.N.Dist3}}(p, center), {{.Radius}});
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"Radius", n.Literal(r),
		),
		EntrypointName: entrypointName,
	}
}

func Rect3DSolid(n Numerics, sideLengths Vec3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect3d_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
					let halfSize = {{.N.Scale3}}({{.SideLengths}}, {{.Half}});
					let absP = {{.N.Abs3}}(p);
					return {{.N.Le}}({{.N.Get3X}}(absP), {{.N.Get3X}}(halfSize)) && {{.N.Le}}({{.N.Get3Y}}(absP), {{.N.Get3Y}}(halfSize)) && {{.N.Le}}({{.N.Get3Z}}(absP), {{.N.Get3Z}}(halfSize));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"SideLengths", sideLengths.WebGPUVec(n),
			"Half", n.Literal(0.5),
		),
		EntrypointName: entrypointName,
	}
}

func Rect3DSDF(n Numerics, sideLengths Vec3) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "rect3d_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> {{.N.Dtype}} {
					let halfSize = {{.N.Scale3}}({{.SideLengths}}, {{.Half}});
					let q = {{.N.Sub3}}({{.N.Abs3}}(p), halfSize);
					let outside = {{.N.Len3}}({{.N.Max3}}(q, {{.Zero3}}));
					let inside = {{.N.Min}}({{.N.Max}}({{.N.Max}}({{.N.Get3X}}(q), {{.N.Get3Y}}(q)), {{.N.Get3Z}}(q)), {{.N.Zero}});
					return {{.N.Sub}}({{.N.Sub}}({{.N.Zero}}, outside), inside);
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"SideLengths", sideLengths.WebGPUVec(n),
			"Half", n.Literal(0.5),
			"Zero3", WGSL("{{.N.Make3}}({{.N.Zero}}, {{.N.Zero}}, {{.N.Zero}})", "N", n.Symbols),
		),
		EntrypointName: entrypointName,
	}
}

func Capsule3DSolid(n Numerics, p1, p2 Vec3, radius float64) ShapeKernel {
	return SDFToSolid(n, Capsule3DSDF(n, p1, p2, radius))
}

// LineJoinSolid creates a solid containing all points within Euclidean
// distance r of any segment, like toolbox3d.LineJoin.
func LineJoinSolid(n Numerics, r float32, lines ...Segment3) ShapeKernel {
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
		Code: WGSL(
			`
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

				fn {{.Entrypoint}}(p_raw: {{.N.Dtype3}}) -> bool {
					let p: vec3<f32> = {{.N.AsFloat3}}(p_raw);
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
			`,
			"N", n.Symbols,
			"SegmentDistance", segmentDistanceName,
			"Entrypoint", entrypointName,
			"NumSegs", len(lines),
			"Segments", bufName,
			"Radius", r,
		),
		EntrypointName: entrypointName,
	}
}

// L1LineJoinSolid creates a solid containing all points within L1 distance r
// of any segment, like toolbox3d.L1LineJoin.
func L1LineJoinSolid(n Numerics, r float32, lines ...Segment3) ShapeKernel {
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
		Code: WGSL(
			`
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

				fn {{.Entrypoint}}(p_raw: {{.N.Dtype3}}) -> bool {
					let p: vec3<f32> = {{.N.AsFloat3}}(p_raw);
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
			"N", n.Symbols,
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

func flattenSegment3s(lines []Segment3) []float32 {
	result := make([]float32, 0, len(lines)*6)
	for _, line := range lines {
		result = append(result, float32(line[0][0]), float32(line[0][1]), float32(line[0][2]))
		result = append(result, float32(line[1][0]), float32(line[1][1]), float32(line[1][2]))
	}
	return result
}

func Capsule3DSDF(n Numerics, p1, p2 Vec3, radius float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "capsule3d_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> {{.N.Dtype}} {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = {{.N.Sub3}}(b, a);
					let pa = {{.N.Sub3}}(p, a);
					let lenSq = {{.N.Dot3}}(ba, ba);
					if ({{.N.Le}}(lenSq, {{.N.Zero}})) {
						return {{.N.Sub}}({{.Radius}}, {{.N.Dist3}}(p, a));
					}
					let h = {{.N.Clamp}}({{.N.Div}}({{.N.Dot3}}(pa, ba), lenSq), {{.N.Zero}}, {{.N.One}});
					return {{.N.Sub}}({{.Radius}}, {{.N.Len3}}({{.N.Sub3}}(pa, {{.N.Scale3}}(ba, h))));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"P1", p1.WebGPUVec(n),
			"P2", p2.WebGPUVec(n),
			"Radius", n.Literal(radius),
		),
		EntrypointName: entrypointName,
	}
}

func CylinderSolid(n Numerics, p1, p2 Vec3, radius float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cylinder_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = {{.N.Sub3}}(b, a);
					let h = {{.N.Len3}}(ba);
					if ({{.N.Le}}(h, {{.N.Zero}})) {
						return {{.N.Le}}({{.N.Dist3}}(p, a), {{.Radius}});
					}
					let axis = {{.N.Scale3}}(ba, {{.N.Div}}({{.N.One}}, h));
					let axial = {{.N.Dot3}}({{.N.Sub3}}(p, a), axis);
					if ({{.N.Lt}}(axial, {{.N.Zero}}) || {{.N.Gt}}(axial, h)) {
						return false;
					}
					let projection = {{.N.Add3}}(a, {{.N.Scale3}}(axis, axial));
					return {{.N.Le}}({{.N.Dist3}}(p, projection), {{.Radius}});
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"P1", p1.WebGPUVec(n),
			"P2", p2.WebGPUVec(n),
			"Radius", n.Literal(radius),
		),
		EntrypointName: entrypointName,
	}
}

func CylinderSDF(n Numerics, p1, p2 Vec3, radius float64) ShapeKernel {
	return ConeSliceSDF(n, p1, p2, radius, radius)
}

func ConeSolid(n Numerics, tip, base Vec3, radius float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
					let tip = {{.Tip}};
					let base = {{.Base}};
					let axisVec = {{.N.Sub3}}(tip, base);
					let h = {{.N.Len3}}(axisVec);
					if ({{.N.Le}}(h, {{.N.Zero}})) {
						return {{.N.Le}}({{.N.Dist3}}(p, tip), {{.Radius}});
					}
					let axis = {{.N.Scale3}}(axisVec, {{.N.Div}}({{.N.One}}, h));
					let axial = {{.N.Dot3}}({{.N.Sub3}}(p, base), axis);
					let radiusFrac = {{.N.Sub}}({{.N.One}}, {{.N.Div}}(axial, h));
					if ({{.N.Lt}}(radiusFrac, {{.N.Zero}}) || {{.N.Gt}}(radiusFrac, {{.N.One}})) {
						return false;
					}
					let projection = {{.N.Add3}}(base, {{.N.Scale3}}(axis, axial));
					return {{.N.Le}}({{.N.Dist3}}(p, projection), {{.N.Mul}}({{.Radius}}, radiusFrac));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"Tip", tip.WebGPUVec(n),
			"Base", base.WebGPUVec(n),
			"Radius", n.Literal(radius),
		),
		EntrypointName: entrypointName,
	}
}

func ConeSDF(n Numerics, tip, base Vec3, radius float64) ShapeKernel {
	return ConeSliceSDF(n, tip, base, 0.0, radius)
}

func ConeSliceSolid(n Numerics, p1, p2 Vec3, r1, r2 float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_slice_solid")
	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> bool {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = {{.N.Sub3}}(b, a);
					let h = {{.N.Len3}}(ba);
					let maxRadius = {{.N.Max}}({{.Radius1}}, {{.Radius2}});
					if ({{.N.Le}}(h, {{.N.Zero}})) {
						return {{.N.Le}}({{.N.Dist3}}(p, a), maxRadius);
					}
					let axis = {{.N.Scale3}}(ba, {{.N.Div}}({{.N.One}}, h));
					let axial = {{.N.Dot3}}({{.N.Sub3}}(p, a), axis);
					let radiusFrac = {{.N.Sub}}({{.N.One}}, {{.N.Div}}(axial, h));
					if ({{.N.Lt}}(radiusFrac, {{.N.Zero}}) || {{.N.Gt}}(radiusFrac, {{.N.One}})) {
						return false;
					}
					let projection = {{.N.Add3}}(a, {{.N.Scale3}}(axis, axial));
					let radius = {{.N.Add}}({{.N.Mul}}({{.Radius1}}, radiusFrac), {{.N.Mul}}({{.Radius2}}, {{.N.Sub}}({{.N.One}}, radiusFrac)));
					return {{.N.Le}}({{.N.Dist3}}(p, projection), radius);
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"P1", p1.WebGPUVec(n),
			"P2", p2.WebGPUVec(n),
			"Radius1", n.Literal(r1),
			"Radius2", n.Literal(r2),
		),
		EntrypointName: entrypointName,
	}
}

func ConeSliceSDF(n Numerics, p1, p2 Vec3, r1, r2 float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "cone_slice_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> {{.N.Dtype}} {
					let a = {{.P1}};
					let b = {{.P2}};
					let ba = {{.N.Sub3}}(b, a);
					let h = {{.N.Len3}}(ba);
					let maxRadius = {{.N.Max}}({{.Radius1}}, {{.Radius2}});
					if ({{.N.Le}}(h, {{.N.Zero}})) {
						return {{.N.Sub}}(maxRadius, {{.N.Dist3}}(p, a));
					}

					let axis = {{.N.Scale3}}(ba, {{.N.Div}}({{.N.One}}, h));
					let pa = {{.N.Sub3}}(p, a);
					let axial = {{.N.Dot3}}(pa, axis);
					let radialVec = {{.N.Sub3}}(pa, {{.N.Scale3}}(axis, axial));
					let radial = {{.N.Len3}}(radialVec);

					var inside = false;
					if ({{.N.Ge}}(axial, {{.N.Zero}}) && {{.N.Le}}(axial, h)) {
						let axialFrac = {{.N.Div}}(axial, h);
						let radius = {{.N.Add}}({{.Radius1}}, {{.N.Mul}}({{.N.Sub}}({{.Radius2}}, {{.Radius1}}), axialFrac));
						inside = {{.N.Le}}(radial, radius);
					}

					let q = {{.N.Make2}}(axial, radial);
					let sideA = {{.N.Make2}}({{.N.Zero}}, {{.Radius1}});
					let sideB = {{.N.Make2}}(h, {{.Radius2}});
					let sideBA = {{.N.Sub2}}(sideB, sideA);
					let sidePASeg = {{.N.Sub2}}(q, sideA);
					let sideLenSq = {{.N.Dot2}}(sideBA, sideBA);
					var sideT = {{.N.Zero}};
					if ({{.N.Gt}}(sideLenSq, {{.N.Zero}})) {
						sideT = {{.N.Clamp}}({{.N.Div}}({{.N.Dot2}}(sidePASeg, sideBA), sideLenSq), {{.N.Zero}}, {{.N.One}});
					}
					let sideDist = {{.N.Len2}}({{.N.Sub2}}(sidePASeg, {{.N.Scale2}}(sideBA, sideT)));

					let cap1Axial = axial;
					var cap1Dist = {{.N.Len2}}(q);
					if ({{.N.Lt}}(radial, {{.Radius1}})) {
						cap1Dist = {{.N.Abs}}(cap1Axial);
					} else {
						cap1Dist = {{.N.Len2}}({{.N.Make2}}(cap1Axial, {{.N.Sub}}(radial, {{.Radius1}})));
					}

					let cap2Axial = {{.N.Sub}}(axial, h);
					var cap2Dist = {{.N.Len2}}({{.N.Make2}}(cap2Axial, radial));
					if ({{.N.Lt}}(radial, {{.Radius2}})) {
						cap2Dist = {{.N.Abs}}(cap2Axial);
					} else {
						cap2Dist = {{.N.Len2}}({{.N.Make2}}(cap2Axial, {{.N.Sub}}(radial, {{.Radius2}})));
					}

					let unsignedDist = {{.N.Min}}(sideDist, {{.N.Min}}(cap1Dist, cap2Dist));
					if (inside) {
						return unsignedDist;
					} else {
						return {{.N.Sub}}({{.N.Zero}}, unsignedDist);
					}
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"P1", p1.WebGPUVec(n),
			"P2", p2.WebGPUVec(n),
			"Radius1", n.Literal(r1),
			"Radius2", n.Literal(r2),
		),
		EntrypointName: entrypointName,
	}
}

func SphereSDF(n Numerics, r float64) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "sphere_sdf")
	return ShapeKernel{
		Kind: SDF3D,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.N.Dtype3}}) -> {{.N.Dtype}} {
					let center = {{.N.Make3}}({{.N.Zero}}, {{.N.Zero}}, {{.N.Zero}});
					return {{.N.Sub}}({{.Radius}}, {{.N.Dist3}}(p, center));
				}
			`,
			"N", n.Symbols,
			"Entrypoint", entrypointName,
			"Radius", n.Literal(r),
		),
		EntrypointName: entrypointName,
	}
}
