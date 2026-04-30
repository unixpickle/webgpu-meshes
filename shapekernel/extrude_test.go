package shapekernel

import (
	"math"
	"math/rand"
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
	kernel := LinearExtrudeSolid(
		Rect2DSolid(Vec2{0.9, 0.5}),
		float32(height),
		center,
		float32(twist),
		Vec2{float32(scale[0]), float32(scale[1])},
	)

	testApproxSolid3D(t, referenceSolid, kernel, 0.03, 0.06)
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
	kernel := LinearExtrudeSDF(Mesh2DSDF(model2d.MarchingSquaresSearch(
		model2d.JoinedSolid{
			&model2d.Circle{Center: model2d.XY(0.25, -0.1), Radius: 0.4},
			model2d.NewRect(model2d.XY(-0.55, -0.2), model2d.XY(0.15, 0.3)),
		},
		0.01,
		8,
	)), float32(height), center)

	testReferenceSDF3D(t, referenceSDF, kernel, 1e-4)
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
	kernel := RevolveSDF(Mesh2DSDF(profileMesh))

	testReferenceSDF3D(t, referenceSDF, kernel, 1e-4)
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
	kernel := RevolveSolidRange(Mesh2DSolid(model2d.MarchingSquaresSearch(profileSolid, 0.01, 8)), float32(angle), float32(start))

	testApproxSolid3D(t, referenceSolid, kernel, 0.03, 0.06)
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
	kernel := RevolveSolid(Mesh2DSolid(model2d.MarchingSquaresSearch(profileSolid, 0.01, 8)))

	testApproxSolid3D(t, referenceSolid, kernel, 0.03, 0.06)
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
	kernel := InsetExtrude(
		Rect2DSDF(Vec2{0.9, 0.5}),
		float32(height),
		center,
		float32(bottom),
		float32(top),
		bottomFn,
		topFn,
	)

	testApproxSolid3D(t, referenceSolid, kernel, 0.03, 0.06)
}

func testApproxSolid3D(t *testing.T, referenceSolid model3d.Solid, kernel ShapeKernel, meshDelta, boundaryEps float64) {
	t.Helper()

	rng := rand.New(rand.NewSource(0))
	referenceMesh := model3d.DualContour(referenceSolid, meshDelta, false, false)
	referenceSDF := model3d.MeshToSDF(referenceMesh)
	center3d := referenceSolid.Min().Mid(referenceSolid.Max())
	extent := referenceSolid.Max().Sub(referenceSolid.Min()).Scale(0.65)

	var inputPoints []Vector
	var expected []bool
	for len(inputPoints) < primitiveTestSamples {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center3d)
		if math.Abs(referenceSDF.SDF(point)) < boundaryEps {
			continue
		}
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, referenceSolid.Contains(point))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}

func testReferenceSDF3D(t *testing.T, referenceSDF model3d.SDF, kernel ShapeKernel, eps float32) {
	t.Helper()

	rng := rand.New(rand.NewSource(0))
	center3d := referenceSDF.Min().Mid(referenceSDF.Max())
	extent := referenceSDF.Max().Sub(referenceSDF.Min()).Scale(0.65)

	var inputPoints []Vector
	var expected []float32
	for i := 0; i < primitiveTestSamples; i++ {
		point := model3d.NewCoord3DRandNorm(rng).Mul(extent).Add(center3d)
		inputPoints = append(inputPoints, Vec3{float32(point.X), float32(point.Y), float32(point.Z)})
		expected = append(expected, float32(referenceSDF.SDF(point)))
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectFloats(t, expected, eps)
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
