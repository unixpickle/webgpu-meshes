package shapekernel

import (
	"fmt"

	"github.com/unixpickle/model3d/model2d"
)

// ArcHullSolid creates a solid kernel for a convex hull of circular arcs.
func ArcHullSolid(h *model2d.ArcHull) ShapeKernel {
	startCenter := Vec2{}
	if h != nil {
		startCenter = Vec2{float32(h.StartCenter.X), float32(h.StartCenter.Y)}
	}

	arcData, segData := flattenArcHull(h)

	ids := IDTracker{}
	arcBufName := genBufferID(&ids, "arc_hull_arcs")
	segBufName := genBufferID(&ids, "arc_hull_segments")
	crossName := genFunctionID(&ids, "cross2d")
	arcContainsName := genFunctionID(&ids, "arc_contains")
	segmentRayScaleName := genFunctionID(&ids, "segment_ray_scale")
	arcRayScaleName := genFunctionID(&ids, "arc_ray_scale")
	entrypointName := genFunctionID(&ids, "arc_hull_solid")

	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Buffers: []Buffer{
			Float32Buffer(arcBufName, func() []float32 {
				return arcData
			}),
			Float32Buffer(segBufName, func() []float32 {
				return segData
			}),
		},
		Code: Dedent(fmt.Sprintf(`
			fn %s(a: vec2<f32>, b: vec2<f32>) -> f32 {
				return a.x * b.y - a.y * b.x;
			}

			fn %s(start: f32, end: f32, theta: f32) -> bool {
				if (start == end) {
					return false;
				}
				if (start > end) {
					return theta <= start && theta > end;
				}
				return theta <= start || theta > end;
			}

			fn %s(origin: vec2<f32>, dir: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
				let v = p2 - p1;
				let denom = %s(dir, v);
				let eps = 1e-6 * length(dir) * length(v);
				if (abs(denom) <= eps) {
					return 1e30;
				}
				let delta = p1 - origin;
				let rayT = %s(delta, v) / denom;
				let segT = %s(delta, dir) / denom;
				if (rayT > 1e-6 && segT >= 0.0 && segT <= 1.0) {
					return rayT;
				}
				return 1e30;
			}

			fn %s(origin: vec2<f32>, dir: vec2<f32>, center: vec2<f32>, radius: f32, start: f32, end: f32) -> f32 {
				if (radius <= 0.0 || start == end) {
					return 1e30;
				}

				let offset = origin - center;
				let a = dot(dir, dir);
				if (a <= 0.0) {
					return 1e30;
				}

				let b = 2.0 * dot(dir, offset);
				let c = dot(offset, offset) - radius * radius;
				let discriminant = b * b - 4.0 * a * c;
				if (discriminant <= 0.0) {
					return 1e30;
				}

				let sqrtDisc = sqrt(discriminant);
				let invDenom = 0.5 / a;
				let t1 = (-b - sqrtDisc) * invDenom;
				let t2 = (-b + sqrtDisc) * invDenom;
				var best = 1e30;

				if (t1 > 1e-6) {
					let point = origin + dir * t1;
					let theta = atan2(point.y - center.y, point.x - center.x);
					if (%s(start, end, theta)) {
						best = t1;
					}
				}
				if (t2 > 1e-6) {
					let point = origin + dir * t2;
					let theta = atan2(point.y - center.y, point.x - center.x);
					if (%s(start, end, theta)) {
						best = min(best, t2);
					}
				}

				return best;
			}

			fn %s(p: vec2<f32>) -> bool {
				let numArcs = %du;
				let numSegs = %du;
				if (numArcs == 0u && numSegs == 0u) {
					return false;
				}

				let startCenter = %s;
				let dir = p - startCenter;
				if (dot(dir, dir) <= 1e-12) {
					return true;
				}

					var bestScale = 1e30;
					for (var i = 0u; i < numArcs; i++) {
							let center = vec2<f32>(%s[i*5], %s[i*5+1]);
							let radius = %s[i*5+2];
							let start = %s[i*5+3];
							let end = %s[i*5+4];
						bestScale = min(bestScale, %s(startCenter, dir, center, radius, start, end));
					}
					for (var i = 0u; i < numSegs; i++) {
							let p1 = vec2<f32>(%s[i*4], %s[i*4+1]);
							let p2 = vec2<f32>(%s[i*4+2], %s[i*4+3]);
						bestScale = min(bestScale, %s(startCenter, dir, p1, p2));
					}

				if (bestScale >= 1e29) {
					return false;
					}
					return bestScale >= 1.0;
				}
				`, crossName, arcContainsName, segmentRayScaleName, crossName, crossName, crossName, arcRayScaleName, arcContainsName, arcContainsName, entrypointName, len(arcData)/5, len(segData)/4, startCenter.WebGPUVec(),
			arcBufName, arcBufName, arcBufName, arcBufName, arcBufName, arcRayScaleName,
			segBufName, segBufName, segBufName, segBufName, segmentRayScaleName)),
		EntrypointName: entrypointName,
	}
}

// ArcHullSDF creates an SDF kernel for a convex hull of circular arcs.
func ArcHullSDF(h *model2d.ArcHull) ShapeKernel {
	solidKernel := ArcHullSolid(h)
	arcBufName := solidKernel.Buffers[0].Name
	segBufName := solidKernel.Buffers[1].Name
	arcContainsName := genFunctionID(&solidKernel.IDs, "arc_contains_dist")
	segmentDistanceName := genFunctionID(&solidKernel.IDs, "segment_distance2d")
	arcDistanceName := genFunctionID(&solidKernel.IDs, "arc_distance2d")
	entrypointName := genFunctionID(&solidKernel.IDs, "arc_hull_sdf")

	arcCount := 0
	segCount := 0
	if h != nil && h.Tree != nil && h.Tree.Root != nil {
		arcCount, segCount = arcHullPrimitiveCounts(h)
	}

	return ShapeKernel{
		Kind:    SDF2D,
		IDs:     solidKernel.IDs,
		Buffers: append([]Buffer{}, solidKernel.Buffers...),
		Code: solidKernel.Code + "\n" + Dedent(fmt.Sprintf(`
			fn %s(start: f32, end: f32, theta: f32) -> bool {
				if (start == end) {
					return false;
				}
				if (start > end) {
					return theta <= start && theta > end;
				}
				return theta <= start || theta > end;
			}

			fn %s(p: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
				let v = p2 - p1;
				let vNormSq = dot(v, v);
				if (vNormSq <= 0.0) {
					return distance(p, p1);
				}
				let t = clamp(dot(p - p1, v) / vNormSq, 0.0, 1.0);
				let closest = p1 + t * v;
				return distance(p, closest);
			}

			fn %s(p: vec2<f32>, center: vec2<f32>, radius: f32, start: f32, end: f32) -> f32 {
				if (radius <= 0.0) {
					return distance(p, center);
				}

				let startPoint = center + radius * vec2<f32>(cos(start), sin(start));
				if (start == end) {
					return distance(p, startPoint);
				}

				let endPoint = center + radius * vec2<f32>(cos(end), sin(end));
				var minDist = min(distance(p, startPoint), distance(p, endPoint));

				let theta = atan2(p.y - center.y, p.x - center.x);
				if (%s(start, end, theta)) {
					minDist = min(minDist, abs(distance(p, center) - radius));
				}

				return minDist;
			}

			fn %s(p: vec2<f32>) -> f32 {
				let numArcs = %du;
				let numSegs = %du;
				if (numArcs == 0u && numSegs == 0u) {
					return 0.0;
				}

					var minDist = 1e30;
					for (var i = 0u; i < numArcs; i++) {
							let center = vec2<f32>(%s[i*5], %s[i*5+1]);
							let radius = %s[i*5+2];
							let start = %s[i*5+3];
							let end = %s[i*5+4];
						minDist = min(minDist, %s(p, center, radius, start, end));
					}
					for (var i = 0u; i < numSegs; i++) {
							let p1 = vec2<f32>(%s[i*4], %s[i*4+1]);
							let p2 = vec2<f32>(%s[i*4+2], %s[i*4+3]);
						minDist = min(minDist, %s(p, p1, p2));
					}

				if (%s(p)) {
					return minDist;
					}
					return -minDist;
				}
				`, arcContainsName, segmentDistanceName, arcDistanceName, arcContainsName, entrypointName, arcCount, segCount,
			arcBufName, arcBufName, arcBufName, arcBufName, arcBufName, arcDistanceName,
			segBufName, segBufName, segBufName, segBufName, segmentDistanceName, solidKernel.EntrypointName)),
		EntrypointName: entrypointName,
	}
}

func flattenArcHull(h *model2d.ArcHull) (arcData, segData []float32) {
	if h == nil || h.Tree == nil || h.Tree.Root == nil {
		return nil, nil
	}

	arcs := []*model2d.ArcHullArc{}
	h.Tree.Iterate(func(arc *model2d.ArcHullArc) bool {
		arcs = append(arcs, arc)
		return true
	})

	arcData = make([]float32, 0, len(arcs)*5)
	for _, arc := range arcs {
		arcData = append(arcData,
			float32(arc.Center.X),
			float32(arc.Center.Y),
			float32(arc.Radius),
			float32(arc.Start),
			float32(arc.End),
		)
	}

	if len(arcs) == 0 {
		return arcData, nil
	}

	prev := arcs[len(arcs)-1]
	for _, arc := range arcs {
		p1 := prev.EndCoord()
		p2 := arc.StartCoord()
		if p1 != p2 {
			segData = append(segData,
				float32(p1.X), float32(p1.Y),
				float32(p2.X), float32(p2.Y),
			)
		}
		prev = arc
	}

	return arcData, segData
}

func arcHullPrimitiveCounts(h *model2d.ArcHull) (arcCount, segCount int) {
	arcData, segData := flattenArcHull(h)
	return len(arcData) / 5, len(segData) / 4
}
