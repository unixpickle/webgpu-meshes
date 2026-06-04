package shapekernel

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model2d"
)

func TestArcHullPrimitive(t *testing.T) {
	shape := randomArcHullForTest()
	testPrimitive2D(t, shape, SmokeFloat32Numerics, ArcHullSolid(SmokeFloat32Numerics, shape), ArcHullSDF(SmokeFloat32Numerics, shape), 2e-4, 5e-4)
}

func TestArcHullSDF(t *testing.T) {
	shape := randomArcHullForTest()
	testPrimitive2DSDF(t, shape, SmokeFloat32Numerics, ArcHullSDF(SmokeFloat32Numerics, shape), 2e-4, 5e-4)
}

func randomArcHullForTest() *model2d.ArcHull {
	rng := rand.New(rand.NewSource(1))
	const count = 20

	circles := make([]*model2d.Circle, 0, count+4)
	for i := 0; i < count; i++ {
		theta := 2 * math.Pi * float64(i) / count
		theta += rng.NormFloat64() * 0.08

		radius := 0.8 + 0.25*rng.Float64()
		center := model2d.NewCoordPolar(theta, radius)
		center = center.Add(model2d.NewCoordRandNorm(rng).Scale(0.04))

		circleRadius := 0.05 + 0.18*rng.Float64()
		if i%5 == 0 {
			circleRadius = 0.0
		}

		circles = append(circles, &model2d.Circle{
			Center: center,
			Radius: circleRadius,
		})
	}

	for i := 0; i < 4; i++ {
		circles = append(circles, &model2d.Circle{
			Center: model2d.NewCoordRandNorm(rng).Scale(0.25),
			Radius: 0.04 + 0.06*rng.Float64(),
		})
	}

	return model2d.NewArcHull(circles)
}
