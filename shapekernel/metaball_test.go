package shapekernel

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
)

func TestMetaball2DFieldScale(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	circle := &model2d.Circle{
		Center: model2d.XY(1.0, 2.0),
		Radius: 0.3,
	}
	scale := model2d.XY(0.5, 1.0)

	kernel := Scale(SmokeFloat32Numerics,
		Translate(SmokeFloat32Numerics,
			SDFToMetaball(SmokeFloat32Numerics, CircleSDF(SmokeFloat32Numerics, circle.Radius)),
			Vec2{circle.Center.X, circle.Center.Y},
		),
		Vec2{scale.X, scale.Y},
	)
	expectedMB := model2d.VecScaleMetaball(circle, scale)

	center := expectedMB.Min().Mid(expectedMB.Max())
	extent := expectedMB.Max().Sub(expectedMB.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec2{point.X, point.Y})
		expected = append(expected, float32(expectedMB.MetaballField(point)))
	}
	vals := ExecuteShapeKernel(t, kernelToNative(SmokeFloat32Numerics, kernel), inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}

func TestMetaballSolid2D(t *testing.T) {
	c1 := &model2d.Circle{Center: model2d.XY(-0.7, 0.4), Radius: 0.45}
	c2 := &model2d.Circle{Center: model2d.XY(0.9, -0.2), Radius: 0.35}
	falloff := GaussianMetaballFalloffFunc(SmokeFloat32Numerics)
	kernel := MetaballSolid(SmokeFloat32Numerics,
		falloff,
		0.5,
		Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, CircleSDF(SmokeFloat32Numerics, c1.Radius)), Vec2{c1.Center.X, c1.Center.Y}),
		Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, CircleSDF(SmokeFloat32Numerics, c2.Radius)), Vec2{c2.Center.X, c2.Center.Y}),
	)
	expectedSolid := model2d.MetaballSolid(model2d.GaussianMetaballFalloffFunc, 0.5, c1, c2)

	testApproxSolid2D(t, expectedSolid, SmokeFloat32Numerics, kernel, 0.01, 0.02)
}

func TestWeightedMetaballSolid2D(t *testing.T) {
	c1 := &model2d.Circle{Center: model2d.XY(-0.7, 0.4), Radius: 0.45}
	c2 := &model2d.Circle{Center: model2d.XY(0.9, -0.2), Radius: 0.35}
	weights := []float64{1.25, -0.4}
	kernel := WeightedMetaballSolid(SmokeFloat32Numerics,
		QuarticMetaballFalloffFunc(SmokeFloat32Numerics),
		0.5,
		[]ShapeKernel{
			Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, CircleSDF(SmokeFloat32Numerics, c1.Radius)), Vec2{c1.Center.X, c1.Center.Y}),
			Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, CircleSDF(SmokeFloat32Numerics, c2.Radius)), Vec2{c2.Center.X, c2.Center.Y}),
		},
		weights,
	)
	expectedSolid := model2d.WeightedMetaballSolid(model2d.QuarticMetaballFalloffFunc,
		0.5,
		[]model2d.Metaball{c1, c2},
		weights,
	)

	testApproxSolid2D(t, expectedSolid, SmokeFloat32Numerics, kernel, 0.01, 0.02)
}

func TestMetaball3DFieldScale(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sphere := &model3d.Sphere{
		Center: model3d.XYZ(1.0, 2.0, -0.3),
		Radius: 0.3,
	}
	scale := model3d.XYZ(0.25, 0.5, 1.0)

	kernel := Scale(SmokeFloat32Numerics,
		Translate(SmokeFloat32Numerics,
			SDFToMetaball(SmokeFloat32Numerics, SphereSDF(SmokeFloat32Numerics, sphere.Radius)),
			Vec3{sphere.Center.X, sphere.Center.Y, sphere.Center.Z},
		),
		Vec3{scale.X, scale.Y, scale.Z},
	)
	expectedMB := model3d.VecScaleMetaball(sphere, scale)

	center := expectedMB.Min().Mid(expectedMB.Max())
	extent := expectedMB.Max().Sub(expectedMB.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec3{point.X, point.Y, point.Z})
		expected = append(expected, float32(expectedMB.MetaballField(point)))
	}
	vals := ExecuteShapeKernel(t, kernelToNative(SmokeFloat32Numerics, kernel), inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}

func TestMetaballSolid3D(t *testing.T) {
	s1 := &model3d.Sphere{Center: model3d.XYZ(-0.7, 0.4, 0.1), Radius: 0.45}
	s2 := &model3d.Sphere{Center: model3d.XYZ(0.9, -0.2, -0.5), Radius: 0.35}
	falloff := WyvillMetaballFalloffFunc(SmokeFloat32Numerics, 0.75)
	kernel := MetaballSolid(SmokeFloat32Numerics,
		falloff,
		0.5,
		Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, SphereSDF(SmokeFloat32Numerics, s1.Radius)), Vec3{s1.Center.X, s1.Center.Y, s1.Center.Z}),
		Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, SphereSDF(SmokeFloat32Numerics, s2.Radius)), Vec3{s2.Center.X, s2.Center.Y, s2.Center.Z}),
	)
	expectedSolid := model3d.MetaballSolid(model3d.WyvillMetaballFalloffFunc(0.75), 0.5, s1, s2)

	testApproxSolid3D(t, expectedSolid, SmokeFloat32Numerics, kernel, 0.03, 0.06)
}

func TestWeightedMetaballSolid3D(t *testing.T) {
	s1 := &model3d.Sphere{Center: model3d.XYZ(-0.7, 0.4, 0.1), Radius: 0.45}
	s2 := &model3d.Sphere{Center: model3d.XYZ(0.9, -0.2, -0.5), Radius: 0.35}
	weights := []float64{1.25, -0.4}
	kernel := WeightedMetaballSolid(SmokeFloat32Numerics,
		QuarticMetaballFalloffFunc(SmokeFloat32Numerics),
		0.5,
		[]ShapeKernel{
			Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, SphereSDF(SmokeFloat32Numerics, s1.Radius)), Vec3{s1.Center.X, s1.Center.Y, s1.Center.Z}),
			Translate(SmokeFloat32Numerics, SDFToMetaball(SmokeFloat32Numerics, SphereSDF(SmokeFloat32Numerics, s2.Radius)), Vec3{s2.Center.X, s2.Center.Y, s2.Center.Z}),
		},
		weights,
	)
	expectedSolid := model3d.WeightedMetaballSolid(model3d.QuarticMetaballFalloffFunc,
		0.5,
		[]model3d.Metaball{s1, s2},
		weights,
	)

	testApproxSolid3D(t, expectedSolid, SmokeFloat32Numerics, kernel, 0.03, 0.06)
}
