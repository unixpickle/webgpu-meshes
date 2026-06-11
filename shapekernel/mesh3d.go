package shapekernel

import "github.com/unixpickle/model3d/model3d"

// Mesh3DSolid creates a solid using the even-odd rule to determine if points
// are within a given triangle mesh.
func Mesh3DSolid(n Numerics, m *model3d.Mesh) ShapeKernel {
	bvh := newMesh3DBVH(m)
	numNodes := len(bvh.NodeData) / 4

	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "mesh3d_solid")
	nodeMinName := genFunctionID(&ids, "mesh3d_node_min")
	nodeMaxName := genFunctionID(&ids, "mesh3d_node_max")
	rayBoundsName := genFunctionID(&ids, "mesh3d_ray_bounds")
	triangleHitName := genFunctionID(&ids, "mesh3d_triangle_hit")
	rayCastName := genFunctionID(&ids, "mesh3d_ray_cast")
	triBufName := genBufferID(&ids, "triangles")
	nodeBoundsBufName := genBufferID(&ids, "node_bounds")
	nodeDataBufName := genBufferID(&ids, "node_data")

	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Buffers: []Buffer{
			Float32Buffer(triBufName, func() []float32 {
				return bvh.Triangles
			}),
			Float32Buffer(nodeBoundsBufName, func() []float32 {
				return bvh.NodeBounds
			}),
			Uint32Buffer(nodeDataBufName, func() []uint32 {
				return bvh.NodeData
			}),
		},
		Code: WGSL(
			`
				fn {{.NodeMin}}(nodeIdx: u32) -> vec3<f32> {
					let offset = nodeIdx * 6u;
					return vec3<f32>({{.NodeBounds}}[offset], {{.NodeBounds}}[offset+1u], {{.NodeBounds}}[offset+2u]);
				}

				fn {{.NodeMax}}(nodeIdx: u32) -> vec3<f32> {
					let offset = nodeIdx * 6u;
					return vec3<f32>({{.NodeBounds}}[offset+3u], {{.NodeBounds}}[offset+4u], {{.NodeBounds}}[offset+5u]);
				}

				fn {{.RayBounds}}(origin: vec3<f32>, dir: vec3<f32>, minVal: vec3<f32>, maxVal: vec3<f32>) -> bool {
					var minFrac = -1e30;
					var maxFrac = 1e30;
					for (var axis = 0; axis < 3; axis++) {
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

				fn {{.TriangleHit}}(origin: vec3<f32>, dir: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> vec2<f32> {
					let v1 = p2 - p1;
					let v2 = p3 - p1;
					let cross1 = cross(dir, v2);
					let det = dot(cross1, v1);
					let eps = 1e-6 * length(v1) * length(cross1);
					if (abs(det) <= eps) {
						return vec2<f32>(0.0, 0.0);
					}
					let invDet = 1.0 / det;
					let o = origin - p1;
					let bary1 = invDet * dot(o, cross1);
					let cross2 = cross(o, v1);
					let bary2 = invDet * dot(dir, cross2);
					let scale = invDet * dot(v2, cross2);
					let hit = bary1 >= 0.0 && bary2 >= 0.0 && bary1 + bary2 <= 1.0 && scale >= 0.0;
					let edgeFraction = min(
						min(abs(bary1), abs(bary2)),
						min(abs(1.0 - bary1 - bary2), abs(scale)),
					);
					return vec2<f32>(select(0.0, 1.0, hit), edgeFraction);
				}

				fn {{.RayCast}}(p: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
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
						let triStart = {{.NodeData}}[nodeDataOffset];
						let triCount = {{.NodeData}}[nodeDataOffset+1u];
						let skipIndex = {{.NodeData}}[nodeDataOffset+3u];
						let minVal = {{.NodeMin}}(nodeIdx);
						let maxVal = {{.NodeMax}}(nodeIdx);
						if (!{{.RayBounds}}(p, dir, minVal, maxVal)) {
							nodeIdx = skipIndex;
							continue;
						}

						if (triCount > 0u) {
							for (var i = 0u; i < triCount; i++) {
								let triIdx = triStart + i;
								let triOffset = triIdx * 9u;
								let p1 = vec3<f32>({{.Triangles}}[triOffset], {{.Triangles}}[triOffset+1u], {{.Triangles}}[triOffset+2u]);
								let p2 = vec3<f32>({{.Triangles}}[triOffset+3u], {{.Triangles}}[triOffset+4u], {{.Triangles}}[triOffset+5u]);
								let p3 = vec3<f32>({{.Triangles}}[triOffset+6u], {{.Triangles}}[triOffset+7u], {{.Triangles}}[triOffset+8u]);
								let hitResult = {{.TriangleHit}}(p, dir, p1, p2, p3);
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

				fn {{.Entrypoint}}(p_raw: {{.N.Dtype3}}) -> bool {
					let p = {{.N.AsFloat3}}(p_raw);
					let first = {{.RayCast}}(p, vec3<f32>(0.5224892708603626, 0.10494477243214506, 0.43558938446126527));
					let second = {{.RayCast}}(p, vec3<f32>(0.10494477243214506, 0.43558938446126527, 0.5224892708603626));
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
			"TriangleHit", triangleHitName,
			"RayCast", rayCastName,
			"Entrypoint", entrypointName,
			"N", n.Symbols,
			"NumNodes", numNodes,
			"NodeData", nodeDataBufName,
			"Triangles", triBufName,
		),
		EntrypointName: entrypointName,
	}
}

// Mesh3DSDF creates an SDF by combining the triangle distance with the
// inside/outside test from Mesh3DSolid.
func Mesh3DSDF(n Numerics, m *model3d.Mesh) ShapeKernel {
	bvh := newMesh3DBVH(m)
	solidKernel := Mesh3DSolid(n, m)
	numNodes := len(solidKernel.Buffers[2].Constructor()) / 4
	numTris := len(solidKernel.Buffers[0].Constructor()) / 9
	queueSize := bvh.Height
	if queueSize < 1 {
		queueSize = 1
	}
	triBufName := solidKernel.Buffers[0].Name
	nodeBoundsBufName := solidKernel.Buffers[1].Name
	nodeDataBufName := solidKernel.Buffers[2].Name
	nodeMinName := genFunctionID(&solidKernel.IDs, "mesh3d_sdf_node_min")
	nodeMaxName := genFunctionID(&solidKernel.IDs, "mesh3d_sdf_node_max")
	pointBoundsDistName := genFunctionID(&solidKernel.IDs, "mesh3d_point_bounds_dist_sq")
	segmentDistanceName := genFunctionID(&solidKernel.IDs, "segment_distance3d")
	triangleDistanceName := genFunctionID(&solidKernel.IDs, "triangle_distance3d")
	entrypointName := genFunctionID(&solidKernel.IDs, "mesh3d_sdf")

	return ShapeKernel{
		Kind:    SDF3D,
		IDs:     solidKernel.IDs,
		Buffers: append([]Buffer{}, solidKernel.Buffers...),
		Code: solidKernel.Code + "\n" + WGSL(
			`
				fn {{.NodeMin}}(nodeIdx: u32) -> vec3<f32> {
					let offset = nodeIdx * 6u;
					return vec3<f32>({{.NodeBounds}}[offset], {{.NodeBounds}}[offset+1u], {{.NodeBounds}}[offset+2u]);
				}

				fn {{.NodeMax}}(nodeIdx: u32) -> vec3<f32> {
					let offset = nodeIdx * 6u;
					return vec3<f32>({{.NodeBounds}}[offset+3u], {{.NodeBounds}}[offset+4u], {{.NodeBounds}}[offset+5u]);
				}

				fn {{.PointBoundsDist}}(p: vec3<f32>, minVal: vec3<f32>, maxVal: vec3<f32>) -> f32 {
					var distSq = 0.0;
					for (var axis = 0; axis < 3; axis++) {
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

				fn {{.TriangleDistance}}(p: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> f32 {
					let ab = p2 - p1;
					let ac = p3 - p1;
					let ap = p - p1;
					let d1 = dot(ab, ap);
					let d2 = dot(ac, ap);
					if (d1 <= 0.0 && d2 <= 0.0) {
						return distance(p, p1);
					}

					let bp = p - p2;
					let d3 = dot(ab, bp);
					let d4 = dot(ac, bp);
					if (d3 >= 0.0 && d4 <= d3) {
						return distance(p, p2);
					}

					let vc = d1 * d4 - d3 * d2;
					if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
						let v = d1 / (d1 - d3);
						return distance(p, p1 + v * ab);
					}

					let cp = p - p3;
					let d5 = dot(ab, cp);
					let d6 = dot(ac, cp);
					if (d6 >= 0.0 && d5 <= d6) {
						return distance(p, p3);
					}

					let vb = d5 * d2 - d1 * d6;
					if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
						let w = d2 / (d2 - d6);
						return distance(p, p1 + w * ac);
					}

					let va = d3 * d6 - d5 * d4;
					if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
						let bc = p3 - p2;
						let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
						return distance(p, p2 + w * bc);
					}

					let n = cross(ab, ac);
					let nLen = length(n);
					if (nLen <= 1e-12) {
						return min(
							min({{.SegmentDistance}}(p, p1, p2), {{.SegmentDistance}}(p, p2, p3)),
							{{.SegmentDistance}}(p, p3, p1),
						);
					}
					return abs(dot(ap, n)) / nLen;
				}

				fn {{.Entrypoint}}(p_raw: {{.N.Dtype3}}) -> {{.N.Dtype}} {
					let p = {{.N.AsFloat3}}(p_raw);
					let numNodes = {{.NumNodes}}u;
					let numTris = {{.NumTris}}u;
					if (numNodes == 0u || numTris == 0u) {
						return {{.N.FromFloat}}(0.0);
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
						let triStart = {{.NodeData}}[nodeDataOffset];
						let triCount = {{.NodeData}}[nodeDataOffset+1u];
						if (triCount > 0u) {
							for (var i = 0u; i < triCount; i++) {
								let triIdx = triStart + i;
								let triOffset = triIdx * 9u;
								let p1 = vec3<f32>({{.Triangles}}[triOffset], {{.Triangles}}[triOffset+1u], {{.Triangles}}[triOffset+2u]);
								let p2 = vec3<f32>({{.Triangles}}[triOffset+3u], {{.Triangles}}[triOffset+4u], {{.Triangles}}[triOffset+5u]);
								let p3 = vec3<f32>({{.Triangles}}[triOffset+6u], {{.Triangles}}[triOffset+7u], {{.Triangles}}[triOffset+8u]);
								let dist = {{.TriangleDistance}}(p, p1, p2, p3);
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

					let pSolid = {{.N.Make3}}({{.N.FromFloat}}(p.x), {{.N.FromFloat}}(p.y), {{.N.FromFloat}}(p.z));
					if ({{.Solid}}(pSolid)) {
						return {{.N.FromFloat}}(minDist);
					}
					return {{.N.FromFloat}}(-minDist);
				}
			`,
			"NodeMin", nodeMinName,
			"NodeBounds", nodeBoundsBufName,
			"NodeMax", nodeMaxName,
			"PointBoundsDist", pointBoundsDistName,
			"SegmentDistance", segmentDistanceName,
			"TriangleDistance", triangleDistanceName,
			"Entrypoint", entrypointName,
			"N", n.Symbols,
			"NumNodes", numNodes,
			"NumTris", numTris,
			"QueueSize", queueSize,
			"NodeData", nodeDataBufName,
			"Triangles", triBufName,
			"Solid", solidKernel.EntrypointName,
		),
		EntrypointName: entrypointName,
	}
}
