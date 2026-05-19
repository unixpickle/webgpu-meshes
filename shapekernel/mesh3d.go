package shapekernel

import (
	"fmt"

	"github.com/unixpickle/model3d/model3d"
)

// Mesh3DSolid creates a solid using the even-odd rule to determine if points
// are within a given triangle mesh.
func Mesh3DSolid(m *model3d.Mesh) ShapeKernel {
	bvh := newMesh3DBVH(m)
	numNodes := len(bvh.NodeData) / 4

	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "mesh3d_solid")
	nodeMinName := genFunctionID(&ids, "mesh3d_node_min")
	nodeMaxName := genFunctionID(&ids, "mesh3d_node_max")
	rayBoundsName := genFunctionID(&ids, "mesh3d_ray_bounds")
	triangleHitName := genFunctionID(&ids, "mesh3d_triangle_hit")
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
		Code: Dedent(fmt.Sprintf(`
			fn %s(nodeIdx: u32) -> vec3<f32> {
				let offset = nodeIdx * 6u;
				return vec3<f32>(%s[offset], %s[offset+1u], %s[offset+2u]);
			}

			fn %s(nodeIdx: u32) -> vec3<f32> {
				let offset = nodeIdx * 6u;
				return vec3<f32>(%s[offset+3u], %s[offset+4u], %s[offset+5u]);
			}

			fn %s(origin: vec3<f32>, dir: vec3<f32>, minVal: vec3<f32>, maxVal: vec3<f32>) -> bool {
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

			fn %s(origin: vec3<f32>, dir: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> bool {
				let v1 = p2 - p1;
				let v2 = p3 - p1;
				let cross1 = cross(dir, v2);
				let det = dot(cross1, v1);
				let eps = 1e-6 * length(v1) * length(cross1);
				if (abs(det) <= eps) {
					return false;
				}
				let invDet = 1.0 / det;
				let o = origin - p1;
				let bary1 = invDet * dot(o, cross1);
				if (bary1 < 0.0 || bary1 > 1.0) {
					return false;
				}
				let cross2 = cross(o, v1);
				let bary2 = invDet * dot(dir, cross2);
				if (bary2 < 0.0 || bary1 + bary2 > 1.0) {
					return false;
				}
				let scale = invDet * dot(v2, cross2);
				return scale >= 0.0;
			}

			fn %s(p: vec3<f32>) -> bool {
				let numNodes = %du;
				if (numNodes == 0u) {
					return false;
				}

				let dir = vec3<f32>(0.5224892708603626, 0.10494477243214506, 0.43558938446126527);
				var numIntersections = 0u;
				var nodeIdx = 0u;
				loop {
					if (nodeIdx >= numNodes) {
						break;
					}

					let nodeDataOffset = nodeIdx * 4u;
					let triStart = %s[nodeDataOffset];
					let triCount = %s[nodeDataOffset+1u];
					let skipIndex = %s[nodeDataOffset+3u];
					let minVal = %s(nodeIdx);
					let maxVal = %s(nodeIdx);
					if (!%s(p, dir, minVal, maxVal)) {
						nodeIdx = skipIndex;
						continue;
					}

					if (triCount > 0u) {
						for (var i = 0u; i < triCount; i++) {
							let triIdx = triStart + i;
							let triOffset = triIdx * 9u;
							let p1 = vec3<f32>(%s[triOffset], %s[triOffset+1u], %s[triOffset+2u]);
							let p2 = vec3<f32>(%s[triOffset+3u], %s[triOffset+4u], %s[triOffset+5u]);
							let p3 = vec3<f32>(%s[triOffset+6u], %s[triOffset+7u], %s[triOffset+8u]);
							if (%s(p, dir, p1, p2, p3)) {
								numIntersections += 1u;
							}
						}
					}

					nodeIdx += 1u;
				}
				return (numIntersections & 1u) == 1u;
			}
		`, nodeMinName, nodeBoundsBufName, nodeBoundsBufName, nodeBoundsBufName,
			nodeMaxName, nodeBoundsBufName, nodeBoundsBufName, nodeBoundsBufName,
			rayBoundsName, triangleHitName, entrypointName, numNodes,
			nodeDataBufName, nodeDataBufName, nodeDataBufName, nodeMinName, nodeMaxName, rayBoundsName,
			triBufName, triBufName, triBufName, triBufName, triBufName, triBufName, triBufName, triBufName, triBufName,
			triangleHitName)),
		EntrypointName: entrypointName,
	}
}

// Mesh3DSDF creates an SDF by combining the triangle distance with the
// inside/outside test from Mesh3DSolid.
func Mesh3DSDF(m *model3d.Mesh) ShapeKernel {
	bvh := newMesh3DBVH(m)
	solidKernel := Mesh3DSolid(m)
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
		Code: solidKernel.Code + "\n" + Dedent(fmt.Sprintf(`
			fn %s(nodeIdx: u32) -> vec3<f32> {
				let offset = nodeIdx * 6u;
				return vec3<f32>(%s[offset], %s[offset+1u], %s[offset+2u]);
			}

			fn %s(nodeIdx: u32) -> vec3<f32> {
				let offset = nodeIdx * 6u;
				return vec3<f32>(%s[offset+3u], %s[offset+4u], %s[offset+5u]);
			}

			fn %s(p: vec3<f32>, minVal: vec3<f32>, maxVal: vec3<f32>) -> f32 {
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

			fn %s(p: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> f32 {
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
						min(%s(p, p1, p2), %s(p, p2, p3)),
						%s(p, p3, p1),
					);
				}
				return abs(dot(ap, n)) / nLen;
			}

			fn %s(p: vec3<f32>) -> f32 {
				let numNodes = %du;
				let numTris = %du;
				if (numNodes == 0u || numTris == 0u) {
					return 0.0;
				}

				var minDist = 1e30;
				var minDistSq = 1e30;
				var pendingNodes: array<u32, %du>;
				var pendingDists: array<f32, %du>;
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
					let triStart = %s[nodeDataOffset];
					let triCount = %s[nodeDataOffset+1u];
					if (triCount > 0u) {
						for (var i = 0u; i < triCount; i++) {
							let triIdx = triStart + i;
							let triOffset = triIdx * 9u;
							let p1 = vec3<f32>(%s[triOffset], %s[triOffset+1u], %s[triOffset+2u]);
							let p2 = vec3<f32>(%s[triOffset+3u], %s[triOffset+4u], %s[triOffset+5u]);
							let p3 = vec3<f32>(%s[triOffset+6u], %s[triOffset+7u], %s[triOffset+8u]);
							let dist = %s(p, p1, p2, p3);
							if (dist < minDist) {
								minDist = dist;
								minDistSq = dist * dist;
							}
						}
						hasCurrent = false;
						continue;
					}

					let leftNode = currentNode + 1u;
					let rightNode = %s[nodeDataOffset+2u];
					let leftDistSq = %s(p, %s(leftNode), %s(leftNode));
					let rightDistSq = %s(p, %s(rightNode), %s(rightNode));

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

					if (farDistSq <= minDistSq && pendingCount < %du) {
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

				if (%s(p)) {
					return minDist;
				}
				return -minDist;
			}
		`, nodeMinName, nodeBoundsBufName, nodeBoundsBufName, nodeBoundsBufName,
			nodeMaxName, nodeBoundsBufName, nodeBoundsBufName, nodeBoundsBufName,
			pointBoundsDistName, segmentDistanceName, triangleDistanceName,
			segmentDistanceName, segmentDistanceName, segmentDistanceName,
			entrypointName, numNodes, numTris, queueSize, queueSize,
			nodeDataBufName, nodeDataBufName,
			triBufName, triBufName, triBufName, triBufName, triBufName, triBufName, triBufName, triBufName, triBufName,
			triangleDistanceName,
			nodeDataBufName, pointBoundsDistName, nodeMinName, nodeMaxName, pointBoundsDistName, nodeMinName, nodeMaxName,
			queueSize, solidKernel.EntrypointName)),
		EntrypointName: entrypointName,
	}
}
