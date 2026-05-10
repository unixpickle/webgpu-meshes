package shapekernel

import (
	"fmt"

	"github.com/unixpickle/model3d/model3d"
)

// Mesh3DSolid creates a solid using the even-odd rule to determine if points
// are within a given triangle mesh.
func Mesh3DSolid(m *model3d.Mesh) ShapeKernel {
	// For now, we simply load all of the triangles into a buffer
	// and check a ray collision with each one.
	bufFn := func() []float32 {
		tris := m.TriangleSlice()
		result := make([]float32, 0, len(tris)*9)
		for _, tri := range tris {
			for _, c := range tri {
				result = append(result, float32(c.X), float32(c.Y), float32(c.Z))
			}
		}
		return result
	}

	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "mesh3d_solid")
	bufName := genBufferID(&ids, "triangles")

	return ShapeKernel{
		Kind: Solid3D,
		IDs:  ids,
		Buffers: []Buffer{
			{
				Name:        bufName,
				Constructor: bufFn,
			},
		},
		Code: Dedent(fmt.Sprintf(`
			fn %s(p: vec3<f32>) -> bool {
				// Count ray-triangle intersections.
				let numTris = %du;
				let dir = vec3<f32>(0.5224892708603626, 0.10494477243214506, 0.43558938446126527);
				var numIntersections = 0u;
				for (var i = 0u; i < numTris; i++) {
					let p1 = vec3<f32>(%s[i*9], %s[i*9+1], %s[i*9+2]);
					let p2 = vec3<f32>(%s[i*9+3], %s[i*9+4], %s[i*9+5]);
					let p3 = vec3<f32>(%s[i*9+6], %s[i*9+7], %s[i*9+8]);
					let v1 = p2 - p1;
					let v2 = p3 - p1;
					let cross1 = cross(dir, v2);
					let det = dot(cross1, v1);
					let eps = 1e-6 * length(v1) * length(cross1);
					if (abs(det) > eps) {
						let invDet = 1.0 / det;
						let o = p - p1;
						let bary1 = invDet * dot(o, cross1);
						if (bary1 >= 0.0 && bary1 <= 1.0) {
							let cross2 = cross(o, v1);
							let bary2 = invDet * dot(dir, cross2);
							if (bary2 >= 0.0 && bary1 + bary2 <= 1.0) {
								let scale = invDet * dot(v2, cross2);
								if (scale >= 0.0) {
									numIntersections += 1u;
								}
							}
						}
					}
				}
				return (numIntersections & 1u) == 1u;
			}
		`, entrypointName, m.NumTriangles(), bufName, bufName, bufName, bufName, bufName, bufName, bufName, bufName, bufName)),
		EntrypointName: entrypointName,
	}
}

// Mesh3DSDF creates an SDF by combining the triangle distance with the
// inside/outside test from Mesh3DSolid.
func Mesh3DSDF(m *model3d.Mesh) ShapeKernel {
	solidKernel := Mesh3DSolid(m)
	bufName := solidKernel.Buffers[0].Name
	segmentDistanceName := genFunctionID(&solidKernel.IDs, "segment_distance3d")
	triangleDistanceName := genFunctionID(&solidKernel.IDs, "triangle_distance3d")
	entrypointName := genFunctionID(&solidKernel.IDs, "mesh3d_sdf")

	return ShapeKernel{
		Kind:    SDF3D,
		IDs:     solidKernel.IDs,
		Buffers: append([]Buffer{}, solidKernel.Buffers...),
		Code: solidKernel.Code + "\n" + Dedent(fmt.Sprintf(`
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
				let numTris = %du;
				if (numTris == 0u) {
					return 0.0;
				}

				var minDist = 1e30;
				for (var i = 0u; i < numTris; i++) {
					let p1 = vec3<f32>(%s[i*9], %s[i*9+1], %s[i*9+2]);
					let p2 = vec3<f32>(%s[i*9+3], %s[i*9+4], %s[i*9+5]);
					let p3 = vec3<f32>(%s[i*9+6], %s[i*9+7], %s[i*9+8]);
					minDist = min(minDist, %s(p, p1, p2, p3));
				}

				if (%s(p)) {
					return minDist;
				} else {
					return -minDist;
				}
			}
		`, segmentDistanceName, triangleDistanceName, segmentDistanceName, segmentDistanceName, segmentDistanceName, entrypointName, m.NumTriangles(), bufName, bufName, bufName, bufName, bufName, bufName, bufName, bufName, bufName, triangleDistanceName, solidKernel.EntrypointName)),
		EntrypointName: entrypointName,
	}
}
