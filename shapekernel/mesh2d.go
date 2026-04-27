package shapekernel

import (
	"fmt"

	"github.com/unixpickle/model3d/model2d"
)

// Mesh2DSolid creates a solid using the even-odd rule to determine if points
// are within a given segment mesh.
func Mesh2DSolid(m2 *model2d.Mesh) ShapeKernel {
	// For now, we simply load all of the segments into a buffer
	// and check a ray collision with each one.
	bufFn := func() []float32 {
		segs := m2.SegmentSlice()
		result := make([]float32, 0, len(segs)*4)
		for _, s := range segs {
			result = append(result, float32(s[0].X))
			result = append(result, float32(s[0].Y))
			result = append(result, float32(s[1].X))
			result = append(result, float32(s[1].Y))
		}
		return result
	}

	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "mesh2d_solid")
	bufName := genBufferID(&ids, "segments")

	return ShapeKernel{
		Kind: Solid2D,
		IDs:  ids,
		Buffers: []Buffer{
			{
				Name:        bufName,
				Constructor: bufFn,
			},
		},
		Code: Dedent(fmt.Sprintf(`
			fn %s(p: vec2<f32>) -> bool {
				// Count ray-segment intersections.
				let numSegs = %du;
				let dir = vec2<f32>(0.5224892708603626, 0.10494477243214506);
				var numIntersections = 0u;
				for (var i = 0u; i < numSegs; i++) {
					let p1 = vec2<f32>(%s[i*4], %s[i*4+1]);
					let p2 = vec2<f32>(%s[i*4+2], %s[i*4+3]);
					let v = p2 - p1;
					let det = v.x * dir.y - v.y * dir.x;
					let eps = 1e-5 * length(v) * length(dir);
					if (abs(det) > eps) {
						let rhs = p - p1;

						// inverse([a b; c d]) = (1/det) * [ d -b; -c a ]
						let result = vec2<f32>(
							( dir.y * rhs.x - dir.x * rhs.y) / det,
							(-v.y   * rhs.x + v.x   * rhs.y) / det
						);

						let segT = result.x;
						let rayT = result.y;

						// Intersection if it lands on the segment and forward on the ray.
						if (segT >= 0.0 && segT < 1.0 && rayT >= 0.0) {
							numIntersections += 1u;
						}
					}
				}
				return (numIntersections & 1u) == 1u;
			}
		`, entrypointName, m2.NumSegments(), bufName, bufName, bufName, bufName)),
		EntrypointName: entrypointName,
	}
}

// Mesh2DSDF creates an SDF by combining the segment distance with the
// inside/outside test from Mesh2DSolid.
func Mesh2DSDF(m2 *model2d.Mesh) ShapeKernel {
	solidKernel := Mesh2DSolid(m2)
	bufName := solidKernel.Buffers[0].Name
	entrypointName := genFunctionID(&solidKernel.IDs, "mesh2d_sdf")

	return ShapeKernel{
		Kind:    SDF2D,
		IDs:     solidKernel.IDs,
		Buffers: append([]Buffer{}, solidKernel.Buffers...),
		Code: solidKernel.Code + "\n" + Dedent(fmt.Sprintf(`
			fn %s(p: vec2<f32>) -> f32 {
				let numSegs = %du;
				if (numSegs == 0u) {
					return 0.0;
				}

				var minDist = 1e30;
				for (var i = 0u; i < numSegs; i++) {
					let p1 = vec2<f32>(%s[i*4], %s[i*4+1]);
					let p2 = vec2<f32>(%s[i*4+2], %s[i*4+3]);
					let v = p2 - p1;
					let vNormSq = dot(v, v);
					var t = 0.0;
					if (vNormSq > 0.0) {
						t = clamp(dot(p - p1, v) / vNormSq, 0.0, 1.0);
					}
					let closest = p1 + t * v;
					minDist = min(minDist, distance(p, closest));
				}

				if (%s(p)) {
					return minDist;
				} else {
					return -minDist;
				}
			}
		`, entrypointName, m2.NumSegments(), bufName, bufName, bufName, bufName, solidKernel.EntrypointName)),
		EntrypointName: entrypointName,
	}
}
