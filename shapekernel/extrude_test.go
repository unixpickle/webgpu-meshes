package shapekernel

import (
	"math"
	"testing"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/toolbox3d"
)

func TestLinearExtrudeSolid(t *testing.T) {
	rect := model2d.NewRect(model2d.XY(-0.45, -0.25), model2d.XY(0.45, 0.25))
	height := float64(1.4)
	center := true
	twist := 1.1
	scale := [2]float64{0.7, 1.35}

	referenceSolid := linearExtrudeReference(rect, height, center, twist, scale)
	kernel := LinearExtrudeSolid(SmokeFloat32Numerics,
		Rect2DSolid(SmokeFloat32Numerics, Vec2{0.9, 0.5}),
		float32(height),
		center,
		float32(twist),
		Vec2{scale[0], scale[1]},
	)

	testApproxSolid3D(t, referenceSolid, SmokeFloat32Numerics, kernel, 0.03, 0.06)
}

func TestLinearExtrudeSDF(t *testing.T) {
	height := 1.4
	center := true
	z0, z1 := linearExtrudeReferenceZBounds(height, center)
	profile := model2d.MeshToSDF(model2d.MarchingSquaresSearch(
		model2d.JoinedSolid{
			&model2d.Circle{Center: model2d.XY(0.25, -0.1), Radius: 0.4},
			model2d.NewRect(model2d.XY(-0.55, -0.2), model2d.XY(0.15, 0.3)),
		},
		0.01,
		8,
	))
	referenceSDF := model3d.ProfileSDF(profile, z0, z1)
	kernel := LinearExtrudeSDF(SmokeFloat32Numerics, Mesh2DSDF(SmokeFloat32Numerics, model2d.MarchingSquaresSearch(
		model2d.JoinedSolid{
			&model2d.Circle{Center: model2d.XY(0.25, -0.1), Radius: 0.4},
			model2d.NewRect(model2d.XY(-0.55, -0.2), model2d.XY(0.15, 0.3)),
		},
		0.01,
		8,
	)), float32(height), center)

	testSDFKernel3D(t, referenceSDF, SmokeFloat32Numerics, kernel, 1e-4)
}

func TestRevolveSDF(t *testing.T) {
	profileSolid := model2d.JoinedSolid{
		&model2d.Circle{Center: model2d.XY(0.35, -0.1), Radius: 0.28},
		&model2d.Circle{Center: model2d.XY(-0.22, 0.18), Radius: 0.19},
		model2d.NewRect(model2d.XY(-0.08, -0.42), model2d.XY(0.16, 0.05)),
	}
	profileMesh := model2d.MarchingSquaresSearch(profileSolid, 0.01, 8)
	profileSDF := model2d.MeshToSDF(profileMesh)
	referenceSDF := model3d.RevolveSDF(profileSDF)
	kernel := RevolveSDF(SmokeFloat32Numerics, Mesh2DSDF(SmokeFloat32Numerics, profileMesh))

	testSDFKernel3D(t, referenceSDF, SmokeFloat32Numerics, kernel, 1e-4)
}

func TestRevolveSolidRange(t *testing.T) {
	profileSolid := model2d.JoinedSolid{
		&model2d.Circle{Center: model2d.XY(0.35, -0.1), Radius: 0.28},
		&model2d.Circle{Center: model2d.XY(-0.22, 0.18), Radius: 0.19},
		model2d.NewRect(model2d.XY(-0.08, -0.42), model2d.XY(0.16, 0.05)),
	}
	angle := 1.6
	start := -0.7
	referenceSolid, err := model3d.RevolveSolidRange(profileSolid, angle, start)
	if err != nil {
		t.Fatal(err)
	}
	kernel := RevolveSolidRange(SmokeFloat32Numerics, Mesh2DSolid(SmokeFloat32Numerics, model2d.MarchingSquaresSearch(profileSolid, 0.01, 8)), float32(angle), float32(start))

	testApproxSolid3D(t, referenceSolid, SmokeFloat32Numerics, kernel, 0.03, 0.06)
}

func TestRevolveSolid(t *testing.T) {
	profileSolid := model2d.JoinedSolid{
		&model2d.Circle{Center: model2d.XY(0.35, -0.1), Radius: 0.28},
		&model2d.Circle{Center: model2d.XY(-0.22, 0.18), Radius: 0.19},
		model2d.NewRect(model2d.XY(-0.08, -0.42), model2d.XY(0.16, 0.05)),
	}
	referenceSolid, err := model3d.RevolveSolidRange(profileSolid, 2*math.Pi, 0)
	if err != nil {
		t.Fatal(err)
	}
	kernel := RevolveSolid(SmokeFloat32Numerics, Mesh2DSolid(SmokeFloat32Numerics, model2d.MarchingSquaresSearch(profileSolid, 0.01, 8)))

	testApproxSolid3D(t, referenceSolid, SmokeFloat32Numerics, kernel, 0.03, 0.06)
}

func TestInsetExtrude(t *testing.T) {
	rect := model2d.NewRect(model2d.XY(-0.45, -0.25), model2d.XY(0.45, 0.25))
	height := float64(1.4)
	center := true
	bottom := 0.14
	top := -0.11
	bottomFn := InsetExtrudeFillet
	topFn := InsetExtrudeChamfer
	z0, z1 := linearExtrudeReferenceZBounds(height, center)

	referenceSolid := toolbox3d.Extrude(
		rect,
		z0,
		z1,
		referenceInsetExtrudeFunc(bottom, top, bottomFn, topFn),
	)
	kernel := InsetExtrude(SmokeFloat32Numerics,
		Rect2DSDF(SmokeFloat32Numerics, Vec2{0.9, 0.5}),
		float32(height),
		center,
		float32(bottom),
		float32(top),
		bottomFn,
		topFn,
	)

	testApproxSolid3D(t, referenceSolid, SmokeFloat32Numerics, kernel, 0.03, 0.06)
}

func linearExtrudeReference(s model2d.Solid, height float64, center bool, twist float64, scale [2]float64) model3d.Solid {
	if height < 0 {
		height = -height
	}
	z0, z1 := linearExtrudeReferenceZBounds(height, center)
	min2 := s.Min()
	max2 := s.Max()
	maxScale := math.Max(math.Abs(scale[0]), math.Abs(scale[1]))
	r := maxCornerRadius2D(min2, max2) * maxScale
	min := model3d.XYZ(-r, -r, z0)
	max := model3d.XYZ(r, r, z1)
	return model3d.CheckedFuncSolid(min, max, func(c model3d.Coord3D) bool {
		if c.Z < z0 || c.Z > z1 {
			return false
		}
		t := 0.0
		if height > 0 {
			t = (c.Z - z0) / height
		}
		x, y, ok := inverseExtrudeTransform(c.X, c.Y, t, twist, scale)
		if !ok {
			return false
		}
		return s.Contains(model2d.XY(x, y))
	})
}

func linearExtrudeReferenceZBounds(height float64, center bool) (float64, float64) {
	z0 := 0.0
	z1 := height
	if center {
		z0 = -height / 2
		z1 = height / 2
	}
	return z0, z1
}

func inverseExtrudeTransform(x, y, t, twist float64, scale [2]float64) (float64, float64, bool) {
	sx := 1 + t*(scale[0]-1)
	sy := 1 + t*(scale[1]-1)
	if sx == 0 || sy == 0 {
		return 0, 0, false
	}
	angle := twist * t
	cosA := math.Cos(angle)
	sinA := math.Sin(angle)
	rx := x*cosA - y*sinA
	ry := x*sinA + y*cosA
	return rx / sx, ry / sy, true
}

func maxCornerRadius2D(min, max model2d.Coord) float64 {
	radii := []float64{
		model2d.XY(min.X, min.Y).Norm(),
		model2d.XY(min.X, max.Y).Norm(),
		model2d.XY(max.X, min.Y).Norm(),
		model2d.XY(max.X, max.Y).Norm(),
	}
	maxRadius := radii[0]
	for _, r := range radii[1:] {
		if r > maxRadius {
			maxRadius = r
		}
	}
	return maxRadius
}

func referenceInsetExtrudeFunc(bottom, top float64, bottomFn, topFn InsetFunction) toolbox3d.InsetFunc {
	return toolbox3d.InsetFuncSum(
		referenceInsetExtrudeSideFunc(bottomFn, bottom, true),
		referenceInsetExtrudeSideFunc(topFn, top, false),
	)
}

func referenceInsetExtrudeSideFunc(kind InsetFunction, radius float64, bottom bool) toolbox3d.InsetFunc {
	switch kind {
	case InsetExtrudeChamfer:
		fn := &toolbox3d.ChamferInsetFunc{Outwards: radius < 0}
		if bottom {
			fn.BottomRadius = math.Abs(radius)
		} else {
			fn.TopRadius = math.Abs(radius)
		}
		return fn
	case InsetExtrudeFillet:
		fn := &toolbox3d.FilletInsetFunc{Outwards: radius < 0}
		if bottom {
			fn.BottomRadius = math.Abs(radius)
		} else {
			fn.TopRadius = math.Abs(radius)
		}
		return fn
	default:
		panic(`inset extrude function must be "chamfer" or "fillet"`)
	}
}
