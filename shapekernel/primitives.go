package shapekernel

import (
	"fmt"
)

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
