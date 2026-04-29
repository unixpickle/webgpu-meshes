package shapekernel

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
)

const primitiveTestSamples = 1024

type primitive2D interface {
	Min() model2d.Coord
	Max() model2d.Coord
	Contains(model2d.Coord) bool
	SDF(model2d.Coord) float64
}

type primitive3D interface {
	Min() model3d.Coord3D
	Max() model3d.Coord3D
	Contains(model3d.Coord3D) bool
	SDF(model3d.Coord3D) float64
}

func testPrimitive2D(t *testing.T, shape primitive2D, solidKernel, sdfKernel ShapeKernel, boundaryEps, sdfEps float32) {
	t.Helper()

	rng := rand.New(rand.NewSource(0))
	center := shape.Min().Mid(shape.Max())
	extent := shape.Max().Sub(shape.Min()).Scale(0.65)

	var sdfInputs []Vector
	var sdfExpected []float32
	for i := 0; i < primitiveTestSamples; i++ {
		point := model2d.NewCoordRandNorm(rng).Mul(extent).Add(center)
		sdfInputs = append(sdfInputs, Vec2{float32(point.X), float32(point.Y)})
		sdfExpected = append(sdfExpected, float32(shape.SDF(point)))
	}
	vals := ExecuteShapeKernel(t, sdfKernel, sdfInputs...)
	vals.ExpectFloats(t, sdfExpected, sdfEps)

	var solidInputs []Vector
	var solidExpected []bool
	for len(solidInputs) < primitiveTestSamples {
		point := model2d.NewCoordRandNorm(rng).Mul(extent).Add(center)
		if math.Abs(shape.SDF(point)) < float64(boundaryEps) {
			continue
		}
		solidInputs = append(solidInputs, Vec2{float32(point.X), float32(point.Y)})
		solidExpected = append(solidExpected, shape.Contains(point))
	}
	vals = ExecuteShapeKernel(t, solidKernel, solidInputs...)
	vals.ExpectBools(t, solidExpected)
}

func testPrimitive3D(t *testing.T, shape primitive3D, solidKernel, sdfKernel ShapeKernel, boundaryEps, sdfEps float32) {
	t.Helper()

	rng := rand.New(rand.NewSource(0))
	center := shape.Min().Mid(shape.Max())
	extent := shape.Max().Sub(shape.Min()).Scale(0.65)

	var sdfInputs []Vector
	var sdfExpected []float32
	for i := 0; i < primitiveTestSamples; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center)
		sdfInputs = append(sdfInputs, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		sdfExpected = append(sdfExpected, float32(shape.SDF(point)))
	}
	vals := ExecuteShapeKernel(t, sdfKernel, sdfInputs...)
	vals.ExpectFloats(t, sdfExpected, sdfEps)

	var solidInputs []Vector
	var solidExpected []bool
	for len(solidInputs) < primitiveTestSamples {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center)
		if math.Abs(shape.SDF(point)) < float64(boundaryEps) {
			continue
		}
		solidInputs = append(solidInputs, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		solidExpected = append(solidExpected, shape.Contains(point))
	}
	vals = ExecuteShapeKernel(t, solidKernel, solidInputs...)
	vals.ExpectBools(t, solidExpected)
}

func TestCirclePrimitive(t *testing.T) {
	radius := float32(0.61)
	shape := &model2d.Circle{Radius: float64(radius)}
	testPrimitive2D(t, shape, CircleSolid(radius), CircleSDF(radius), 1e-4, 1e-4)
}

func TestRect2DPrimitive(t *testing.T) {
	sideLengths := Vec2{0.7, 1.3}
	shape := model2d.NewRect(
		model2d.XY(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2),
		model2d.XY(float64(sideLengths[0])/2, float64(sideLengths[1])/2),
	)
	testPrimitive2D(t, shape, Rect2DSolid(sideLengths), Rect2DSDF(sideLengths), 1e-4, 1e-4)
}

func TestCapsule2DPrimitive(t *testing.T) {
	p1 := Vec2{-0.8, 0.35}
	p2 := Vec2{0.55, -0.45}
	radius := float32(0.28)
	shape := &model2d.Capsule{
		P1:     model2d.XY(float64(p1[0]), float64(p1[1])),
		P2:     model2d.XY(float64(p2[0]), float64(p2[1])),
		Radius: float64(radius),
	}
	testPrimitive2D(t, shape, Capsule2DSolid(p1, p2, radius), Capsule2DSDF(p1, p2, radius), 1e-4, 1e-4)
}

func TestSpherePrimitive(t *testing.T) {
	radius := float32(0.61)
	shape := &model3d.Sphere{Radius: float64(radius)}
	testPrimitive3D(t, shape, SphereSolid(radius), SphereSDF(radius), 1e-4, 1e-4)
}

func TestRect3DPrimitive(t *testing.T) {
	sideLengths := Vec3{0.7, 1.3, 0.9}
	shape := model3d.NewRect(
		model3d.XYZ(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2, -float64(sideLengths[2])/2),
		model3d.XYZ(float64(sideLengths[0])/2, float64(sideLengths[1])/2, float64(sideLengths[2])/2),
	)
	testPrimitive3D(t, shape, Rect3DSolid(sideLengths), Rect3DSDF(sideLengths), 1e-4, 1e-4)
}

func TestCapsule3DPrimitive(t *testing.T) {
	p1 := Vec3{-0.8, 0.35, 0.2}
	p2 := Vec3{0.55, -0.45, 0.9}
	radius := float32(0.28)
	shape := &model3d.Capsule{
		P1:     model3d.XYZ(float64(p1[0]), float64(p1[1]), float64(p1[2])),
		P2:     model3d.XYZ(float64(p2[0]), float64(p2[1]), float64(p2[2])),
		Radius: float64(radius),
	}
	testPrimitive3D(t, shape, Capsule3DSolid(p1, p2, radius), Capsule3DSDF(p1, p2, radius), 1e-4, 1e-4)
}

func TestCylinderPrimitive(t *testing.T) {
	p1 := Vec3{-0.8, 0.35, 0.2}
	p2 := Vec3{0.55, -0.45, 0.9}
	radius := float32(0.28)
	shape := &model3d.Cylinder{
		P1:     model3d.XYZ(float64(p1[0]), float64(p1[1]), float64(p1[2])),
		P2:     model3d.XYZ(float64(p2[0]), float64(p2[1]), float64(p2[2])),
		Radius: float64(radius),
	}
	testPrimitive3D(t, shape, CylinderSolid(p1, p2, radius), CylinderSDF(p1, p2, radius), 2e-4, 2e-4)
}

func TestConePrimitive(t *testing.T) {
	tip := Vec3{-0.6, 0.4, -0.3}
	base := Vec3{0.8, -0.2, 0.9}
	radius := float32(0.52)
	shape := &model3d.Cone{
		Tip:    model3d.XYZ(float64(tip[0]), float64(tip[1]), float64(tip[2])),
		Base:   model3d.XYZ(float64(base[0]), float64(base[1]), float64(base[2])),
		Radius: float64(radius),
	}
	testPrimitive3D(t, shape, ConeSolid(tip, base, radius), ConeSDF(tip, base, radius), 5e-4, 5e-4)
}

func TestConeSlicePrimitive(t *testing.T) {
	p1 := Vec3{-0.6, 0.4, -0.3}
	p2 := Vec3{0.8, -0.2, 0.9}
	r1 := float32(0.22)
	r2 := float32(0.52)
	shape := &model3d.ConeSlice{
		P1: model3d.XYZ(float64(p1[0]), float64(p1[1]), float64(p1[2])),
		P2: model3d.XYZ(float64(p2[0]), float64(p2[1]), float64(p2[2])),
		R1: float64(r1),
		R2: float64(r2),
	}
	testPrimitive3D(t, shape, ConeSliceSolid(p1, p2, r1, r2), ConeSliceSDF(p1, p2, r1, r2), 5e-4, 5e-4)
}
