package shapekernel

import (
	"math"
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

func TestSubtractSolid2D(t *testing.T) {
	positive := &model2d.Circle{Radius: 0.85}
	negative := model2d.TransformSolid(
		&model2d.Translate{Offset: model2d.XY(0.35, 0.0)},
		&model2d.Circle{Radius: 0.5},
	)
	referenceSolid := model2d.Subtract(positive, negative)
	referenceSDF := model2d.SubtractSDF(
		positive,
		model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.35, 0.0)}, &model2d.Circle{Radius: 0.5}),
	)
	testPrimitive2D(
		t,
		solidSDF2D{solid: referenceSolid, sdf: referenceSDF},
		SubtractSolid(
			CircleSolid(0.85),
			Translate(CircleSolid(0.5), Vec2{0.35, 0.0}),
		),
		SubtractSDF(
			CircleSDF(0.85),
			Translate(CircleSDF(0.5), Vec2{0.35, 0.0}),
		),
		1e-4,
		1e-4,
	)
}

func TestSubtractSDF2D(t *testing.T) {
	positive := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(-0.15, 0.1)}, &model2d.Circle{Radius: 0.85})
	negative := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.3, -0.05)}, &model2d.Circle{Radius: 0.45})
	referenceSDF := model2d.SubtractSDF(positive, negative)
	testPrimitive2DSDF(
		t,
		solidSDF2DFromSDF(referenceSDF),
		SubtractSDF(
			Translate(CircleSDF(0.85), Vec2{-0.15, 0.1}),
			Translate(CircleSDF(0.45), Vec2{0.3, -0.05}),
		),
		1e-4,
		1e-4,
	)
}

func clipReference2D(shape model2d.SDF, min, max model2d.Coord) solidSDF2D {
	boundsMin, boundsMax := shape.Min(), shape.Max()
	if !math.IsInf(min.X, -1) {
		boundsMin.X = math.Max(boundsMin.X, min.X)
	}
	if !math.IsInf(min.Y, -1) {
		boundsMin.Y = math.Max(boundsMin.Y, min.Y)
	}
	if !math.IsInf(max.X, 1) {
		boundsMax.X = math.Min(boundsMax.X, max.X)
	}
	if !math.IsInf(max.Y, 1) {
		boundsMax.Y = math.Min(boundsMax.Y, max.Y)
	}
	sdf := model2d.FuncSDF(boundsMin, boundsMax, func(c model2d.Coord) float64 {
		return math.Min(shape.SDF(c), clipBoundsSDF2D(c, min, max))
	})
	return solidSDF2D{
		solid: model2d.CheckedFuncSolid(boundsMin, boundsMax, func(c model2d.Coord) bool {
			return sdf.SDF(c) >= 0
		}),
		sdf: sdf,
	}
}

func clipBoundsSDF2D(c, min, max model2d.Coord) float64 {
	outsideX1 := math.Max(min.X-c.X, 0)
	outsideX2 := math.Max(c.X-max.X, 0)
	outsideY1 := math.Max(min.Y-c.Y, 0)
	outsideY2 := math.Max(c.Y-max.Y, 0)
	outside := math.Sqrt(outsideX1*outsideX1 + outsideX2*outsideX2 + outsideY1*outsideY1 + outsideY2*outsideY2)
	if c.X >= min.X && c.X <= max.X && c.Y >= min.Y && c.Y <= max.Y {
		return math.Min(c.X-min.X, math.Min(max.X-c.X, math.Min(c.Y-min.Y, max.Y-c.Y)))
	}
	return -outside
}

func clipReference3D(shape model3d.SDF, min, max model3d.Coord3D) solidSDF3D {
	boundsMin, boundsMax := shape.Min(), shape.Max()
	if !math.IsInf(min.X, -1) {
		boundsMin.X = math.Max(boundsMin.X, min.X)
	}
	if !math.IsInf(min.Y, -1) {
		boundsMin.Y = math.Max(boundsMin.Y, min.Y)
	}
	if !math.IsInf(min.Z, -1) {
		boundsMin.Z = math.Max(boundsMin.Z, min.Z)
	}
	if !math.IsInf(max.X, 1) {
		boundsMax.X = math.Min(boundsMax.X, max.X)
	}
	if !math.IsInf(max.Y, 1) {
		boundsMax.Y = math.Min(boundsMax.Y, max.Y)
	}
	if !math.IsInf(max.Z, 1) {
		boundsMax.Z = math.Min(boundsMax.Z, max.Z)
	}
	sdf := model3d.FuncSDF(boundsMin, boundsMax, func(c model3d.Coord3D) float64 {
		return math.Min(shape.SDF(c), clipBoundsSDF3D(c, min, max))
	})
	return solidSDF3D{
		solid: model3d.CheckedFuncSolid(boundsMin, boundsMax, func(c model3d.Coord3D) bool {
			return sdf.SDF(c) >= 0
		}),
		sdf: sdf,
	}
}

func clipBoundsSDF3D(c, min, max model3d.Coord3D) float64 {
	outsideX1 := math.Max(min.X-c.X, 0)
	outsideX2 := math.Max(c.X-max.X, 0)
	outsideY1 := math.Max(min.Y-c.Y, 0)
	outsideY2 := math.Max(c.Y-max.Y, 0)
	outsideZ1 := math.Max(min.Z-c.Z, 0)
	outsideZ2 := math.Max(c.Z-max.Z, 0)
	outside := math.Sqrt(
		outsideX1*outsideX1 + outsideX2*outsideX2 +
			outsideY1*outsideY1 + outsideY2*outsideY2 +
			outsideZ1*outsideZ1 + outsideZ2*outsideZ2,
	)
	if c.X >= min.X && c.X <= max.X && c.Y >= min.Y && c.Y <= max.Y && c.Z >= min.Z && c.Z <= max.Z {
		return math.Min(
			c.X-min.X,
			math.Min(
				max.X-c.X,
				math.Min(
					c.Y-min.Y,
					math.Min(max.Y-c.Y, math.Min(c.Z-min.Z, max.Z-c.Z)),
				),
			),
		)
	}
	return -outside
}

func TestClipSolid2D(t *testing.T) {
	shape := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.15, -0.1)}, &model2d.Circle{Radius: 0.85})
	min := model2d.XY(-0.2, math.Inf(-1))
	max := model2d.XY(math.Inf(1), 0.45)
	testPrimitive2D(
		t,
		clipReference2D(shape, min, max),
		Clip(
			SDFToSolid(Translate(CircleSDF(0.85), Vec2{0.15, -0.1})),
			Vec2{-0.2, float32(math.Inf(-1))},
			Vec2{float32(math.Inf(1)), 0.45},
		),
		Clip(
			Translate(CircleSDF(0.85), Vec2{0.15, -0.1}),
			Vec2{-0.2, float32(math.Inf(-1))},
			Vec2{float32(math.Inf(1)), 0.45},
		),
		1e-4,
		1e-4,
	)
}

func TestClipSDF2D(t *testing.T) {
	shape := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.15, -0.1)}, &model2d.Circle{Radius: 0.85})
	min := model2d.XY(-0.35, -0.4)
	max := model2d.XY(0.4, 0.55)
	rect := model2d.NewRect(min, max)
	reference := solidSDF2D{
		solid: model2d.CheckedFuncSolid(min.Max(shape.Min()), max.Min(shape.Max()), func(c model2d.Coord) bool {
			return model2d.IntersectSDFs([]model2d.SDF{shape, rect}).SDF(c) >= 0
		}),
		sdf: model2d.IntersectSDFs([]model2d.SDF{shape, rect}),
	}
	testPrimitive2DSDF(
		t,
		reference,
		Clip(
			Translate(CircleSDF(0.85), Vec2{0.15, -0.1}),
			Vec2{-0.35, -0.4},
			Vec2{0.4, 0.55},
		),
		1e-4,
		1e-4,
	)
}

func TestClipSolid3D(t *testing.T) {
	shape := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(0.1, -0.05, 0.2)}, &model3d.Sphere{Radius: 0.9})
	min := model3d.XYZ(-0.4, -0.25, -0.3)
	max := model3d.XYZ(0.55, 0.4, 0.6)
	rect := model3d.NewRect(min, max)
	referenceSDF := model3d.IntersectSDFs([]model3d.SDF{shape, rect})
	testPrimitive3D(
		t,
		solidSDF3D{
			solid: model3d.CheckedFuncSolid(min.Max(shape.Min()), max.Min(shape.Max()), func(c model3d.Coord3D) bool {
				return referenceSDF.SDF(c) >= 0
			}),
			sdf: referenceSDF,
		},
		Clip(
			SDFToSolid(Translate(SphereSDF(0.9), Vec3{0.1, -0.05, 0.2})),
			Vec3{-0.4, -0.25, -0.3},
			Vec3{0.55, 0.4, 0.6},
		),
		Clip(
			Translate(SphereSDF(0.9), Vec3{0.1, -0.05, 0.2}),
			Vec3{-0.4, -0.25, -0.3},
			Vec3{0.55, 0.4, 0.6},
		),
		1e-4,
		1e-4,
	)
}

func TestClipSDF3D(t *testing.T) {
	shape := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(0.1, -0.05, 0.2)}, &model3d.Sphere{Radius: 0.9})
	min := model3d.XYZ(math.Inf(-1), -0.2, -0.15)
	max := model3d.XYZ(0.5, math.Inf(1), 0.7)
	testPrimitive3DSDF(
		t,
		clipReference3D(shape, min, max),
		Clip(
			Translate(SphereSDF(0.9), Vec3{0.1, -0.05, 0.2}),
			Vec3{float32(math.Inf(-1)), -0.2, -0.15},
			Vec3{0.5, float32(math.Inf(1)), 0.7},
		),
		1e-4,
		1e-4,
	)
}

func TestSubtractSolid3D(t *testing.T) {
	positive := &model3d.Sphere{Radius: 0.9}
	negative := model3d.TransformSolid(
		&model3d.Translate{Offset: model3d.XYZ(0.35, 0.0, 0.1)},
		&model3d.Sphere{Radius: 0.5},
	)
	referenceSolid := model3d.Subtract(positive, negative)
	referenceSDF := model3d.SubtractSDF(
		positive,
		model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(0.35, 0.0, 0.1)}, &model3d.Sphere{Radius: 0.5}),
	)
	testPrimitive3D(
		t,
		solidSDF3D{solid: referenceSolid, sdf: referenceSDF},
		SubtractSolid(
			SphereSolid(0.9),
			Translate(SphereSolid(0.5), Vec3{0.35, 0.0, 0.1}),
		),
		SubtractSDF(
			SphereSDF(0.9),
			Translate(SphereSDF(0.5), Vec3{0.35, 0.0, 0.1}),
		),
		1e-4,
		1e-4,
	)
}

func TestSubtractSDF3D(t *testing.T) {
	positive := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(-0.15, 0.1, -0.05)}, &model3d.Sphere{Radius: 0.9})
	negative := model3d.TransformSDF(&model3d.Translate{Offset: model3d.XYZ(0.3, -0.05, 0.2)}, &model3d.Sphere{Radius: 0.45})
	referenceSDF := model3d.SubtractSDF(positive, negative)
	testPrimitive3DSDF(
		t,
		solidSDF3DFromSDF(referenceSDF),
		SubtractSDF(
			Translate(SphereSDF(0.9), Vec3{-0.15, 0.1, -0.05}),
			Translate(SphereSDF(0.45), Vec3{0.3, -0.05, 0.2}),
		),
		1e-4,
		1e-4,
	)
}

func TestUnionSDF2D(t *testing.T) {
	s1 := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(-0.35, 0.1)}, &model2d.Circle{Radius: 0.8})
	s2 := model2d.TransformSDF(&model2d.Translate{Offset: model2d.XY(0.45, -0.2)}, &model2d.Circle{Radius: 0.55})
	referenceSDF := model2d.JoinSDFs([]model2d.SDF{s1, s2})
	testPrimitive2DSDF(
		t,
		solidSDF2DFromSDF(referenceSDF),
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
		solidSDF2DFromSDF(referenceSDF),
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
		solidSDF3DFromSDF(referenceSDF),
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
		solidSDF3DFromSDF(referenceSDF),
		IntersectSDFs([]ShapeKernel{
			Translate(SphereSDF(0.8), Vec3{-0.1, 0.0, 0.15}),
			Translate(SphereSDF(0.8), Vec3{0.35, 0.0, -0.1}),
		}),
		1e-4,
		1e-4,
	)
}
