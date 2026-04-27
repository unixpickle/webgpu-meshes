package shapekernel

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
)

func TestCircleSolid(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	radius := float32(rng.Float64()/2 + 0.25)
	circle := &model2d.Circle{Radius: float64(radius)}
	kernel := CircleSolid(radius)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Scale(1.3 * float64(radius))
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, circle.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func TestCircleSDF(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	radius := float32(rng.Float64()/2 + 0.25)
	circle := &model2d.Circle{Radius: float64(radius)}
	kernel := CircleSDF(radius)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Scale(1.3 * float64(radius))
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, float32(circle.SDF(point)))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}

func TestRect2DSolid(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sideLengths := Vec2{
		float32(rng.Float64() + 0.25),
		float32(rng.Float64() + 0.25),
	}
	rect := model2d.NewRect(
		model2d.XY(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2),
		model2d.XY(float64(sideLengths[0])/2, float64(sideLengths[1])/2),
	)
	kernel := Rect2DSolid(sideLengths)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(model2d.XY(
			float64(sideLengths[0])*1.3,
			float64(sideLengths[1])*1.3,
		))
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, rect.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func TestRect2DSDF(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sideLengths := Vec2{
		float32(rng.Float64() + 0.25),
		float32(rng.Float64() + 0.25),
	}
	rect := model2d.NewRect(
		model2d.XY(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2),
		model2d.XY(float64(sideLengths[0])/2, float64(sideLengths[1])/2),
	)
	kernel := Rect2DSDF(sideLengths)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(model2d.XY(
			float64(sideLengths[0])*1.3,
			float64(sideLengths[1])*1.3,
		))
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, float32(rect.SDF(point)))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}

func TestRect3DSolid(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sideLengths := Vec3{
		float32(rng.Float64() + 0.25),
		float32(rng.Float64() + 0.25),
		float32(rng.Float64() + 0.25),
	}
	rect := model3d.NewRect(
		model3d.XYZ(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2, -float64(sideLengths[2])/2),
		model3d.XYZ(float64(sideLengths[0])/2, float64(sideLengths[1])/2, float64(sideLengths[2])/2),
	)
	kernel := Rect3DSolid(sideLengths)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(model3d.XYZ(
			float64(sideLengths[0])*1.3,
			float64(sideLengths[1])*1.3,
			float64(sideLengths[2])*1.3,
		))
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, rect.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func TestRect3DSDF(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sideLengths := Vec3{
		float32(rng.Float64() + 0.25),
		float32(rng.Float64() + 0.25),
		float32(rng.Float64() + 0.25),
	}
	rect := model3d.NewRect(
		model3d.XYZ(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2, -float64(sideLengths[2])/2),
		model3d.XYZ(float64(sideLengths[0])/2, float64(sideLengths[1])/2, float64(sideLengths[2])/2),
	)
	kernel := Rect3DSDF(sideLengths)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < 1024; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(model3d.XYZ(
			float64(sideLengths[0])*1.3,
			float64(sideLengths[1])*1.3,
			float64(sideLengths[2])*1.3,
		))
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, float32(rect.SDF(point)))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectFloats(t, expected, 1e-4)
}
