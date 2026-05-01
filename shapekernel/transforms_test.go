package shapekernel

import (
	"math"
	"testing"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
)

func TestRotate2D(t *testing.T) {
	sideLengths := Vec2{0.7, 1.3}
	angle := 0.63
	rect := model2d.NewRect(
		model2d.XY(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2),
		model2d.XY(float64(sideLengths[0])/2, float64(sideLengths[1])/2),
	)
	referenceSolid := model2d.RotateSolid(rect, angle)
	referenceSDF := model2d.TransformSDF(model2d.Rotation(angle), rect)
	testPrimitive2D(
		t,
		solidSDF2D{solid: referenceSolid, sdf: referenceSDF},
		Rotate2D(Rect2DSolid(sideLengths), float32(angle)),
		Rotate2D(Rect2DSDF(sideLengths), float32(angle)),
		1e-4,
		1e-4,
	)
}

func TestRotate3D(t *testing.T) {
	sideLengths := Vec3{0.7, 1.3, 0.9}
	axis := model3d.XYZ(1.0, 2.0, -0.5).Normalize()
	angle := 0.47
	rect := model3d.NewRect(
		model3d.XYZ(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2, -float64(sideLengths[2])/2),
		model3d.XYZ(float64(sideLengths[0])/2, float64(sideLengths[1])/2, float64(sideLengths[2])/2),
	)
	referenceSolid := model3d.RotateSolid(rect, axis, angle)
	referenceSDF := model3d.TransformSDF(model3d.Rotation(axis, angle), rect)
	testPrimitive3D(
		t,
		solidSDF3D{solid: referenceSolid, sdf: referenceSDF},
		Rotate3D(
			Rect3DSolid(sideLengths),
			Vec3{float32(axis.X), float32(axis.Y), float32(axis.Z)},
			float32(angle),
		),
		Rotate3D(
			Rect3DSDF(sideLengths),
			Vec3{float32(axis.X), float32(axis.Y), float32(axis.Z)},
			float32(angle),
		),
		1e-4,
		1e-4,
	)
}

func TestMirror2D(t *testing.T) {
	sideLengths := Vec2{0.7, 1.3}
	offset := model2d.XY(0.45, -0.3)
	axis := model2d.XY(1.0, 2.0)
	rect := model2d.NewRect(
		model2d.XY(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2),
		model2d.XY(float64(sideLengths[0])/2, float64(sideLengths[1])/2),
	)
	transform := model2d.JoinedTransform{
		&model2d.Translate{Offset: offset},
		model2d.Mirror(axis),
	}
	testPrimitive2D(
		t,
		solidSDF2D{
			solid: model2d.TransformSolid(transform, rect),
			sdf:   model2d.TransformSDF(transform, rect),
		},
		Mirror2D(
			Translate(Rect2DSolid(sideLengths), Vec2{float32(offset.X), float32(offset.Y)}),
			Vec2{float32(axis.X), float32(axis.Y)},
		),
		Mirror2D(
			Translate(Rect2DSDF(sideLengths), Vec2{float32(offset.X), float32(offset.Y)}),
			Vec2{float32(axis.X), float32(axis.Y)},
		),
		1e-4,
		1e-4,
	)
}

func TestMirror3D(t *testing.T) {
	sideLengths := Vec3{0.7, 1.3, 0.9}
	offset := model3d.XYZ(0.45, -0.3, 0.2)
	axis := model3d.XYZ(1.0, 2.0, -0.5)
	rect := model3d.NewRect(
		model3d.XYZ(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2, -float64(sideLengths[2])/2),
		model3d.XYZ(float64(sideLengths[0])/2, float64(sideLengths[1])/2, float64(sideLengths[2])/2),
	)
	transform := model3d.JoinedTransform{
		&model3d.Translate{Offset: offset},
		model3d.Mirror(axis),
	}
	testPrimitive3D(
		t,
		solidSDF3D{
			solid: model3d.TransformSolid(transform, rect),
			sdf:   model3d.TransformSDF(transform, rect),
		},
		Mirror3D(
			Translate(Rect3DSolid(sideLengths), Vec3{float32(offset.X), float32(offset.Y), float32(offset.Z)}),
			Vec3{float32(axis.X), float32(axis.Y), float32(axis.Z)},
		),
		Mirror3D(
			Translate(Rect3DSDF(sideLengths), Vec3{float32(offset.X), float32(offset.Y), float32(offset.Z)}),
			Vec3{float32(axis.X), float32(axis.Y), float32(axis.Z)},
		),
		1e-4,
		1e-4,
	)
}

func TestScale2D(t *testing.T) {
	sideLengths := Vec2{0.7, 1.3}
	offset := model2d.XY(0.45, -0.3)
	scale := model2d.XY(-1.25, 1.25)
	rect := model2d.NewRect(
		model2d.XY(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2),
		model2d.XY(float64(sideLengths[0])/2, float64(sideLengths[1])/2),
	)
	translatedSDF := model2d.TransformSDF(&model2d.Translate{Offset: offset}, rect)
	expectedSolid := model2d.VecScaleSolid(model2d.TranslateSolid(rect, offset), scale)
	invScale := model2d.XY(1.0/scale.X, 1.0/scale.Y)
	scaleAbs := math.Abs(scale.X)
	expectedSDF := model2d.FuncSDF(expectedSolid.Min(), expectedSolid.Max(), func(c model2d.Coord) float64 {
		return translatedSDF.SDF(c.Mul(invScale)) * scaleAbs
	})
	testPrimitive2D(
		t,
		solidSDF2D{solid: expectedSolid, sdf: expectedSDF},
		Scale(
			Translate(Rect2DSolid(sideLengths), Vec2{float32(offset.X), float32(offset.Y)}),
			Vec2{float32(scale.X), float32(scale.Y)},
		),
		Scale(
			Translate(Rect2DSDF(sideLengths), Vec2{float32(offset.X), float32(offset.Y)}),
			Vec2{float32(scale.X), float32(scale.Y)},
		),
		1e-4,
		1e-4,
	)
}

func TestScale3D(t *testing.T) {
	sideLengths := Vec3{0.7, 1.3, 0.9}
	offset := model3d.XYZ(0.45, -0.3, 0.2)
	scale := model3d.XYZ(-1.25, 1.25, -1.25)
	rect := model3d.NewRect(
		model3d.XYZ(-float64(sideLengths[0])/2, -float64(sideLengths[1])/2, -float64(sideLengths[2])/2),
		model3d.XYZ(float64(sideLengths[0])/2, float64(sideLengths[1])/2, float64(sideLengths[2])/2),
	)
	translatedSDF := model3d.TransformSDF(&model3d.Translate{Offset: offset}, rect)
	expectedSolid := model3d.VecScaleSolid(model3d.TranslateSolid(rect, offset), scale)
	invScale := model3d.XYZ(1.0/scale.X, 1.0/scale.Y, 1.0/scale.Z)
	scaleAbs := math.Abs(scale.X)
	expectedSDF := model3d.FuncSDF(expectedSolid.Min(), expectedSolid.Max(), func(c model3d.Coord3D) float64 {
		return translatedSDF.SDF(c.Mul(invScale)) * scaleAbs
	})
	testPrimitive3D(
		t,
		solidSDF3D{solid: expectedSolid, sdf: expectedSDF},
		Scale(
			Translate(Rect3DSolid(sideLengths), Vec3{float32(offset.X), float32(offset.Y), float32(offset.Z)}),
			Vec3{float32(scale.X), float32(scale.Y), float32(scale.Z)},
		),
		Scale(
			Translate(Rect3DSDF(sideLengths), Vec3{float32(offset.X), float32(offset.Y), float32(offset.Z)}),
			Vec3{float32(scale.X), float32(scale.Y), float32(scale.Z)},
		),
		1e-4,
		1e-4,
	)
}

func TestScaleSDFRequiresUniformAbs(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for non-uniform SDF scale")
		}
	}()
	Scale(Rect2DSDF(Vec2{0.7, 1.3}), Vec2{1.0, 2.0})
}
