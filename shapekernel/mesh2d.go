package shapekernel

import "github.com/unixpickle/model3d/model2d"

// Mesh2DSolid creates a solid using the even-odd rule to determine if points
// are within a given segment mesh.
func Mesh2DSolid(m2 *model2d.Mesh) ShapeKernel {
	bvh := newMesh2DBVH(m2)
	numNodes := len(bvh.NodeData) / 4

	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "mesh2d_solid")
	nodeMinName := genFunctionID(&ids, "mesh2d_node_min")
	nodeMaxName := genFunctionID(&ids, "mesh2d_node_max")
	rayBoundsName := genFunctionID(&ids, "mesh2d_ray_bounds")
	segmentHitName := genFunctionID(&ids, "mesh2d_segment_hit")
	rayCastName := genFunctionID(&ids, "mesh2d_ray_cast")
	segBufName := genBufferID(&ids, "segments")
	nodeBoundsBufName := genBufferID(&ids, "node_bounds")
	nodeDataBufName := genBufferID(&ids, "node_data")

	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Buffers: []Buffer{
			Float32Buffer(segBufName, func() []float32 {
				return bvh.Segments
			}),
			Float32Buffer(nodeBoundsBufName, func() []float32 {
				return bvh.NodeBounds
			}),
			Uint32Buffer(nodeDataBufName, func() []uint32 {
				return bvh.NodeData
			}),
		},
		Code: WGSL(`
			fn {{.NodeMin}}(nodeIdx: u32) -> vec2<f32> {
				let offset = nodeIdx * 4u;
				return vec2<f32>({{.NodeBounds}}[offset], {{.NodeBounds}}[offset+1u]);
			}

			fn {{.NodeMax}}(nodeIdx: u32) -> vec2<f32> {
				let offset = nodeIdx * 4u;
				return vec2<f32>({{.NodeBounds}}[offset+2u], {{.NodeBounds}}[offset+3u]);
			}

			fn {{.RayBounds}}(origin: vec2<f32>, dir: vec2<f32>, minVal: vec2<f32>, maxVal: vec2<f32>) -> bool {
				var minFrac = -1e30;
				var maxFrac = 1e30;
				for (var axis = 0; axis < 2; axis++) {
					let originVal = origin[axis];
					let rate = dir[axis];
					let minBound = minVal[axis];
					let maxBound = maxVal[axis];
					if (rate == 0.0) {
						if (originVal < minBound || originVal > maxBound) {
							return false;
						}
						continue;
					}
					var t1 = (minBound - originVal) / rate;
					var t2 = (maxBound - originVal) / rate;
					if (t1 > t2) {
						let tmp = t1;
						t1 = t2;
						t2 = tmp;
					}
					if (t2 < 0.0) {
						return false;
					}
					minFrac = max(minFrac, t1);
					maxFrac = min(maxFrac, t2);
				}
				return maxFrac >= minFrac && maxFrac >= 0.0;
			}

			fn {{.SegmentHit}}(origin: vec2<f32>, dir: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> vec2<f32> {
				let v = p2 - p1;
				let det = v.x * dir.y - v.y * dir.x;
				let eps = 1e-5 * length(v) * length(dir);
				if (abs(det) <= eps) {
					return vec2<f32>(0.0, 0.0);
				}

				let rhs = origin - p1;
				let result = vec2<f32>(
					(dir.y * rhs.x - dir.x * rhs.y) / det,
					(-v.y * rhs.x + v.x * rhs.y) / det,
				);
				let segT = result.x;
				let rayT = result.y;
				let hit = segT >= 0.0 && segT < 1.0 && rayT >= 0.0;
				let edgeFraction = min(min(abs(segT), abs(segT - 1.0)), abs(rayT));
				return vec2<f32>(select(0.0, 1.0, hit), edgeFraction);
			}

			fn {{.RayCast}}(p: vec2<f32>, dir: vec2<f32>) -> vec2<f32> {
				let numNodes = {{.NumNodes}}u;
				if (numNodes == 0u) {
					return vec2<f32>(0.0, 1e30);
				}

				var numIntersections = 0u;
				var minEdgeFraction = 1e30;
				var nodeIdx = 0u;
				loop {
					if (nodeIdx >= numNodes) {
						break;
					}

					let nodeDataOffset = nodeIdx * 4u;
					let segStart = {{.NodeData}}[nodeDataOffset];
					let segCount = {{.NodeData}}[nodeDataOffset+1u];
					let skipIndex = {{.NodeData}}[nodeDataOffset+3u];
					let minVal = {{.NodeMin}}(nodeIdx);
					let maxVal = {{.NodeMax}}(nodeIdx);
					if (!{{.RayBounds}}(p, -dir, minVal, maxVal)) {
						nodeIdx = skipIndex;
						continue;
					}

					if (segCount > 0u) {
						for (var i = 0u; i < segCount; i++) {
							let segIdx = segStart + i;
							let segOffset = segIdx * 4u;
							let p1 = vec2<f32>({{.Segments}}[segOffset], {{.Segments}}[segOffset+1u]);
							let p2 = vec2<f32>({{.Segments}}[segOffset+2u], {{.Segments}}[segOffset+3u]);
							let hitResult = {{.SegmentHit}}(p, dir, p1, p2);
							if (hitResult.x >= 0.5) {
								numIntersections += 1u;
							}
							minEdgeFraction = min(minEdgeFraction, hitResult.y);
						}
					}

					nodeIdx += 1u;
				}
				return vec2<f32>(f32(numIntersections & 1u), minEdgeFraction);
			}

			fn {{.Entrypoint}}(p: vec2<f32>) -> bool {
				let first = {{.RayCast}}(p, vec2<f32>(0.5224892708603626, 0.10494477243214506));
				let second = {{.RayCast}}(p, vec2<f32>(0.10494477243214506, 0.5224892708603626));
				if (second.y > first.y) {
					return second.x >= 0.5;
				}
				return first.x >= 0.5;
			}
		`,
			"NodeMin", nodeMinName,
			"NodeBounds", nodeBoundsBufName,
			"NodeMax", nodeMaxName,
			"RayBounds", rayBoundsName,
			"SegmentHit", segmentHitName,
			"RayCast", rayCastName,
			"Entrypoint", entrypointName,
			"NumNodes", numNodes,
			"NodeData", nodeDataBufName,
			"Segments", segBufName,
		),
		EntrypointName: entrypointName,
	}
}

// Mesh2DSDF creates an SDF by combining the segment distance with the
// inside/outside test from Mesh2DSolid.
func Mesh2DSDF(m2 *model2d.Mesh) ShapeKernel {
	bvh := newMesh2DBVH(m2)
	solidKernel := Mesh2DSolid(m2)
	numNodes := len(bvh.NodeData) / 4
	numSegs := len(bvh.Segments) / 4
	queueSize := bvh.Height
	if queueSize < 1 {
		queueSize = 1
	}
	segBufName := solidKernel.Buffers[0].Name
	nodeBoundsBufName := solidKernel.Buffers[1].Name
	nodeDataBufName := solidKernel.Buffers[2].Name
	nodeMinName := genFunctionID(&solidKernel.IDs, "mesh2d_sdf_node_min")
	nodeMaxName := genFunctionID(&solidKernel.IDs, "mesh2d_sdf_node_max")
	pointBoundsDistName := genFunctionID(&solidKernel.IDs, "mesh2d_point_bounds_dist_sq")
	segmentDistanceName := genFunctionID(&solidKernel.IDs, "segment_distance2d")
	entrypointName := genFunctionID(&solidKernel.IDs, "mesh2d_sdf")

	return ShapeKernel{
		Kind:    SDF2D,
		IDs:     solidKernel.IDs,
		Buffers: append([]Buffer{}, solidKernel.Buffers...),
		Code: solidKernel.Code + "\n" + WGSL(`
			fn {{.NodeMin}}(nodeIdx: u32) -> vec2<f32> {
				let offset = nodeIdx * 4u;
				return vec2<f32>({{.NodeBounds}}[offset], {{.NodeBounds}}[offset+1u]);
			}

			fn {{.NodeMax}}(nodeIdx: u32) -> vec2<f32> {
				let offset = nodeIdx * 4u;
				return vec2<f32>({{.NodeBounds}}[offset+2u], {{.NodeBounds}}[offset+3u]);
			}

			fn {{.PointBoundsDist}}(p: vec2<f32>, minVal: vec2<f32>, maxVal: vec2<f32>) -> f32 {
				var distSq = 0.0;
				for (var axis = 0; axis < 2; axis++) {
					let value = p[axis];
					let minBound = minVal[axis];
					let maxBound = maxVal[axis];
					if (value < minBound) {
						let delta = minBound - value;
						distSq += delta * delta;
					} else if (value > maxBound) {
						let delta = value - maxBound;
						distSq += delta * delta;
					}
				}
				return distSq;
			}

			fn {{.SegmentDistance}}(p: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
				let v = p2 - p1;
				let vNormSq = dot(v, v);
				var t = 0.0;
				if (vNormSq > 0.0) {
					t = clamp(dot(p - p1, v) / vNormSq, 0.0, 1.0);
				}
				let closest = p1 + t * v;
				return distance(p, closest);
			}

			fn {{.Entrypoint}}(p: vec2<f32>) -> f32 {
				let numNodes = {{.NumNodes}}u;
				let numSegs = {{.NumSegments}}u;
				if (numNodes == 0u || numSegs == 0u) {
					return 0.0;
				}

				var minDist = 1e30;
				var minDistSq = 1e30;
				var pendingNodes: array<u32, {{.QueueSize}}>;
				var pendingDists: array<f32, {{.QueueSize}}>;
				var pendingCount = 0u;
				var currentNode = 0u;
				var currentDistSq = 0.0;
				var hasCurrent = true;

				loop {
					if (!hasCurrent) {
						if (pendingCount == 0u) {
							break;
						}
						pendingCount -= 1u;
						currentNode = pendingNodes[pendingCount];
						currentDistSq = pendingDists[pendingCount];
						hasCurrent = true;
						continue;
					}

					if (currentDistSq > minDistSq) {
						hasCurrent = false;
						continue;
					}

					let nodeDataOffset = currentNode * 4u;
					let segStart = {{.NodeData}}[nodeDataOffset];
					let segCount = {{.NodeData}}[nodeDataOffset+1u];
					if (segCount > 0u) {
						for (var i = 0u; i < segCount; i++) {
							let segIdx = segStart + i;
							let segOffset = segIdx * 4u;
							let p1 = vec2<f32>({{.Segments}}[segOffset], {{.Segments}}[segOffset+1u]);
							let p2 = vec2<f32>({{.Segments}}[segOffset+2u], {{.Segments}}[segOffset+3u]);
							let dist = {{.SegmentDistance}}(p, p1, p2);
							if (dist < minDist) {
								minDist = dist;
								minDistSq = dist * dist;
							}
						}
						hasCurrent = false;
						continue;
					}

					let leftNode = currentNode + 1u;
					let rightNode = {{.NodeData}}[nodeDataOffset+2u];
					let leftDistSq = {{.PointBoundsDist}}(p, {{.NodeMin}}(leftNode), {{.NodeMax}}(leftNode));
					let rightDistSq = {{.PointBoundsDist}}(p, {{.NodeMin}}(rightNode), {{.NodeMax}}(rightNode));

					var nearNode = leftNode;
					var nearDistSq = leftDistSq;
					var farNode = rightNode;
					var farDistSq = rightDistSq;
					if (rightDistSq < leftDistSq) {
						nearNode = rightNode;
						nearDistSq = rightDistSq;
						farNode = leftNode;
						farDistSq = leftDistSq;
					}

					if (farDistSq <= minDistSq && pendingCount < {{.QueueSize}}u) {
						pendingNodes[pendingCount] = farNode;
						pendingDists[pendingCount] = farDistSq;
						pendingCount += 1u;
					}

					if (nearDistSq <= minDistSq) {
						currentNode = nearNode;
						currentDistSq = nearDistSq;
						continue;
					}
					hasCurrent = false;
				}

				if ({{.Solid}}(p)) {
					return minDist;
				}
				return -minDist;
			}
		`,
			"Entrypoint", entrypointName,
			"NodeMin", nodeMinName,
			"NodeBounds", nodeBoundsBufName,
			"NodeMax", nodeMaxName,
			"PointBoundsDist", pointBoundsDistName,
			"SegmentDistance", segmentDistanceName,
			"NumNodes", numNodes,
			"NumSegments", numSegs,
			"QueueSize", queueSize,
			"NodeData", nodeDataBufName,
			"Segments", segBufName,
			"Solid", solidKernel.EntrypointName,
		),
		EntrypointName: entrypointName,
	}
}
