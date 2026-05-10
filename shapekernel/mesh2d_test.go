package shapekernel

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model2d"
)

func testRandomMesh2D(rng *rand.Rand) *model2d.Mesh {
	sourceSolid := model2d.JoinedSolid{}
	for i := 0; i < 15; i++ {
		center := model2d.NewCoordRandNorm(rng)
		sourceSolid = append(
			sourceSolid,
			&model2d.Circle{Center: center, Radius: rng.Float64()/3 + 0.1},
		)
	}
	return model2d.MarchingSquaresSearch(sourceSolid, 0.01, 8)
}

func TestMesh2DSolid(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	mesh := testRandomMesh2D(rng)
	meshSDF := model2d.MeshToSDF(mesh)
	testPrimitive2D(t, solidSDF2DFromSDF(meshSDF), Mesh2DSolid(mesh), Mesh2DSDF(mesh), 0.01, 1e-4)
}

func TestMesh2DSDF(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	mesh := testRandomMesh2D(rng)
	meshSDF := model2d.MeshToSDF(mesh)
	testPrimitive2DSDF(t, solidSDF2DFromSDF(meshSDF), Mesh2DSDF(mesh), 0.01, 1e-4)
}
