package shapekernel

import (
	"fmt"
)

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
