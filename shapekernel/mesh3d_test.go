package shapekernel

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func testRandomMesh3D(rng *rand.Rand) *model3d.Mesh {
	sourceSolid := model3d.JoinedSolid{}
	for i := 0; i < 8; i++ {
		center := model3d.XYZ(
			rng.Float64()*1.2-0.6,
			rng.Float64()*1.2-0.6,
			rng.Float64()*1.2-0.6,
		)
		sourceSolid = append(sourceSolid, &model3d.Sphere{
			Center: center,
			Radius: rng.Float64()/4 + 0.15,
		})
	}
	return model3d.MarchingCubesSearch(sourceSolid, 0.08, 8)
}

func TestMesh3DSolid(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	mesh := testRandomMesh3D(rng)
	meshSDF := model3d.MeshToSDF(mesh)
	testPrimitive3D(t, solidSDF3DFromSDF(meshSDF), Mesh3DSolid(mesh), Mesh3DSDF(mesh), 0.02, 2e-4)
}

func TestMesh3DSDF(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	mesh := testRandomMesh3D(rng)
	meshSDF := model3d.MeshToSDF(mesh)
	testPrimitive3DSDF(t, solidSDF3DFromSDF(meshSDF), Mesh3DSDF(mesh), 0.02, 2e-4)
}
