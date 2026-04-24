package shapekernel

import (
	"testing"
)

func TestUnionSolid(t *testing.T) {
	s1 := SphereSolid(1)
	s2 := Translate(SphereSolid(0.5), Vec3{1, 1, 1})
	joined := UnionSolid(s1, s2)
	vals := ExecuteShapeKernel(t, joined, Vec3{0, 0, 0}, Vec3{1, 1, 1}, Vec3{0.58, 0.58, 0.58}, Vec3{2, 0, 0})
	vals.ExpectBools(t, []bool{true, true, false, false})
}
