package shapekernel

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model2d"
)

func TestMesh2DSolid(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	sourceSolid := model2d.JoinedSolid{}
	for i := 0; i < 15; i++ {
		center := model2d.NewCoordRandNorm(rng)
		sourceSolid = append(
			sourceSolid,
			&model2d.Circle{Center: center, Radius: rng.Float64()/3 + 0.1},
		)
	}
	mesh := model2d.MarchingSquaresSearch(sourceSolid, 0.01, 8)
	kernel := Mesh2DSolid(mesh)
	meshSDF := model2d.MeshToSDF(mesh)

	var inputPoints []Vector
	var expected []bool
	for i := 0; i < 1024; i++ {
		point := model2d.NewCoordRandNorm(rng).Scale(1.3)
		sdf := meshSDF.SDF(point)
		if math.Abs(sdf) < 0.01 {
			i--
			continue
		}
		inputPoints = append(inputPoints, Vec2{float32(point.X), float32(point.Y)})
		expected = append(expected, sdf > 0)
	}
	vals := ExecuteShapeKernel(t, kernel, inputPoints...)
	vals.ExpectBools(t, expected)
}
