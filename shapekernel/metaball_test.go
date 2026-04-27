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

	kernel := Scale(
		Translate(
			SDFToMetaball(CircleSDF(float32(circle.Radius))),
			Vec2{float32(circle.Center.X), float32(circle.Center.Y)},
		),
		Vec2{float32(scale.X), float32(scale.Y)},
	)
	expectedMB := model2d.VecScaleMetaball(circle, scale)

	center := expectedMB.Min().Mid(expectedMB.Max())
	extent := expectedMB.Max().Sub(expectedMB.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, float32(expectedMB.MetaballField(point)))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}

func TestMetaballSolid2D(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	c1 := &model2d.Circle{Center: model2d.XY(-0.7, 0.4), Radius: 0.45}
	c2 := &model2d.Circle{Center: model2d.XY(0.9, -0.2), Radius: 0.35}
	falloff := GaussianMetaballFalloffFunc()
	kernel := MetaballSolid(
		falloff,
		0.5,
		Translate(SDFToMetaball(CircleSDF(float32(c1.Radius))), Vec2{float32(c1.Center.X), float32(c1.Center.Y)}),
		Translate(SDFToMetaball(CircleSDF(float32(c2.Radius))), Vec2{float32(c2.Center.X), float32(c2.Center.Y)}),
	)
	expectedSolid := model2d.MetaballSolid(model2d.GaussianMetaballFalloffFunc, 0.5, c1, c2)

	center := expectedSolid.Min().Mid(expectedSolid.Max())
	extent := expectedSolid.Max().Sub(expectedSolid.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, expectedSolid.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func TestWeightedMetaballSolid2D(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	c1 := &model2d.Circle{Center: model2d.XY(-0.7, 0.4), Radius: 0.45}
	c2 := &model2d.Circle{Center: model2d.XY(0.9, -0.2), Radius: 0.35}
	weights := []float32{1.25, -0.4}
	kernel := WeightedMetaballSolid(
		QuarticMetaballFalloffFunc(),
		0.5,
		[]ShapeKernel{
			Translate(SDFToMetaball(CircleSDF(float32(c1.Radius))), Vec2{float32(c1.Center.X), float32(c1.Center.Y)}),
			Translate(SDFToMetaball(CircleSDF(float32(c2.Radius))), Vec2{float32(c2.Center.X), float32(c2.Center.Y)}),
		},
		weights,
	)
	expectedSolid := model2d.WeightedMetaballSolid(
		model2d.QuarticMetaballFalloffFunc,
		0.5,
		[]model2d.Metaball{c1, c2},
		[]float64{float64(weights[0]), float64(weights[1])},
	)

	center := expectedSolid.Min().Mid(expectedSolid.Max())
	extent := expectedSolid.Max().Sub(expectedSolid.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, expectedSolid.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func TestMetaball3DFieldScale(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sphere := &model3d.Sphere{
		Center: model3d.XYZ(1.0, 2.0, -0.3),
		Radius: 0.3,
	}
	scale := model3d.XYZ(0.25, 0.5, 1.0)

	kernel := Scale(
		Translate(
			SDFToMetaball(SphereSDF(float32(sphere.Radius))),
			Vec3{float32(sphere.Center.X), float32(sphere.Center.Y), float32(sphere.Center.Z)},
		),
		Vec3{float32(scale.X), float32(scale.Y), float32(scale.Z)},
	)
	expectedMB := model3d.VecScaleMetaball(sphere, scale)

	center := expectedMB.Min().Mid(expectedMB.Max())
	extent := expectedMB.Max().Sub(expectedMB.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, float32(expectedMB.MetaballField(point)))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}

func TestMetaballSolid3D(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	s1 := &model3d.Sphere{Center: model3d.XYZ(-0.7, 0.4, 0.1), Radius: 0.45}
	s2 := &model3d.Sphere{Center: model3d.XYZ(0.9, -0.2, -0.5), Radius: 0.35}
	falloff := WyvillMetaballFalloffFunc(0.75)
	kernel := MetaballSolid(
		falloff,
		0.5,
		Translate(SDFToMetaball(SphereSDF(float32(s1.Radius))), Vec3{float32(s1.Center.X), float32(s1.Center.Y), float32(s1.Center.Z)}),
		Translate(SDFToMetaball(SphereSDF(float32(s2.Radius))), Vec3{float32(s2.Center.X), float32(s2.Center.Y), float32(s2.Center.Z)}),
	)
	expectedSolid := model3d.MetaballSolid(model3d.WyvillMetaballFalloffFunc(0.75), 0.5, s1, s2)

	center := expectedSolid.Min().Mid(expectedSolid.Max())
	extent := expectedSolid.Max().Sub(expectedSolid.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, expectedSolid.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func TestWeightedMetaballSolid3D(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	s1 := &model3d.Sphere{Center: model3d.XYZ(-0.7, 0.4, 0.1), Radius: 0.45}
	s2 := &model3d.Sphere{Center: model3d.XYZ(0.9, -0.2, -0.5), Radius: 0.35}
	weights := []float32{1.25, -0.4}
	kernel := WeightedMetaballSolid(
		QuarticMetaballFalloffFunc(),
		0.5,
		[]ShapeKernel{
			Translate(SDFToMetaball(SphereSDF(float32(s1.Radius))), Vec3{float32(s1.Center.X), float32(s1.Center.Y), float32(s1.Center.Z)}),
			Translate(SDFToMetaball(SphereSDF(float32(s2.Radius))), Vec3{float32(s2.Center.X), float32(s2.Center.Y), float32(s2.Center.Z)}),
		},
		weights,
	)
	expectedSolid := model3d.WeightedMetaballSolid(
		model3d.QuarticMetaballFalloffFunc,
		0.5,
		[]model3d.Metaball{s1, s2},
		[]float64{float64(weights[0]), float64(weights[1])},
	)

	center := expectedSolid.Min().Mid(expectedSolid.Max())
	extent := expectedSolid.Max().Sub(expectedSolid.Min()).Scale(1.3)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center)
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, expectedSolid.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}
