package shapekernel

import (
	"testing"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
)

func TestUnionSolid(t *testing.T) {
	s1 := SphereSolid(1)
	s2 := Translate(SphereSolid(0.5), Vec3{1, 1, 1})
	s3 := Translate(SphereSolid(0.5), Vec3{-0.58, -0.58, -0.58})
	joined := UnionSolids([]ShapeKernel{s1, s2, s3})
	vals := ExecuteShapeKernel(
		t,
		joined,
		Vec3{0, 0, 0},
		Vec3{1, 1, 1},
		Vec3{0.58, 0.58, 0.58},
		Vec3{2, 0, 0},
		Vec3{-0.62, -0.62, -0.62},
	)
	vals.ExpectBools(t, []bool{true, true, false, false, true})
}

func TestUnionSDF2D(t *testing.T) {
	s1 := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(-0.35, 0.1)}, &model2d.Circle{Radius: 0.8})
	s2 := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.45, -0.2)}, &model2d.Circle{Radius: 0.55})
	referenceSDF := model2d.JoinSDFs([]model2d.SDF{s1, s2})
	testPrimitive2DSDF(
		t,
		solidSDF2D{
			solid: model2d.CheckedFuncSolid(referenceSDF.Min(), referenceSDF.Max(), func(c model2d.Coord) bool {
				return referenceSDF.SDF(c) >= 0
			}),
			sdf: referenceSDF,
		},
		UnionSDFs([]ShapeKernel{
			Translate(CircleSDF(0.8), Vec2{-0.35, 0.1}),
			Translate(CircleSDF(0.55), Vec2{0.45, -0.2}),
		}),
		1e-4,
		1e-4,
	)
}

func TestIntersectSDF2D(t *testing.T) {
	s1 := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(-0.1, 0.0)}, &model2d.Circle{Radius: 0.8})
	s2 := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.35, 0.0)}, &model2d.Circle{Radius: 0.8})
	referenceSDF := model2d.IntersectSDFs([]model2d.SDF{s1, s2})
	testPrimitive2DSDF(
		t,
		solidSDF2D{
			solid: model2d.CheckedFuncSolid(referenceSDF.Min(), referenceSDF.Max(), func(c model2d.Coord) bool {
				return referenceSDF.SDF(c) >= 0
			}),
			sdf: referenceSDF,
		},
		IntersectSDFs([]ShapeKernel{
			Translate(CircleSDF(0.8), Vec2{-0.1, 0.0}),
			Translate(CircleSDF(0.8), Vec2{0.35, 0.0}),
		}),
		1e-4,
		1e-4,
	)
}

func TestUnionSDF3D(t *testing.T) {
	s1 := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(-0.35, 0.1, -0.2)}, &model3d.Sphere{Radius: 0.8})
	s2 := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(0.45, -0.2, 0.3)}, &model3d.Sphere{Radius: 0.55})
	referenceSDF := model3d.JoinSDFs([]model3d.SDF{s1, s2})
	testPrimitive3DSDF(
		t,
		solidSDF3D{
			solid: model3d.CheckedFuncSolid(referenceSDF.Min(), referenceSDF.Max(), func(c model3d.Coord3D) bool {
				return referenceSDF.SDF(c) >= 0
			}),
			sdf: referenceSDF,
		},
		UnionSDFs([]ShapeKernel{
			Translate(SphereSDF(0.8), Vec3{-0.35, 0.1, -0.2}),
			Translate(SphereSDF(0.55), Vec3{0.45, -0.2, 0.3}),
		}),
		1e-4,
		1e-4,
	)
}

func TestIntersectSDF3D(t *testing.T) {
	s1 := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(-0.1, 0.0, 0.15)}, &model3d.Sphere{Radius: 0.8})
	s2 := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(0.35, 0.0, -0.1)}, &model3d.Sphere{Radius: 0.8})
	referenceSDF := model3d.IntersectSDFs([]model3d.SDF{s1, s2})
	testPrimitive3DSDF(
		t,
		solidSDF3D{
			solid: model3d.CheckedFuncSolid(referenceSDF.Min(), referenceSDF.Max(), func(c model3d.Coord3D) bool {
				return referenceSDF.SDF(c) >= 0
			}),
			sdf: referenceSDF,
		},
		IntersectSDFs([]ShapeKernel{
			Translate(SphereSDF(0.8), Vec3{-0.1, 0.0, 0.15}),
			Translate(SphereSDF(0.8), Vec3{0.35, 0.0, -0.1}),
		}),
		1e-4,
		1e-4,
	)
}
