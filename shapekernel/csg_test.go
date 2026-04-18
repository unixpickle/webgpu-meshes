package shapekernel

import (
	"testing"
)

func TestUnionSolid(t *testing.T) {
	s1 := SphereSolid(1)
	s2 := Translate(SphereSolid(0.5), Vec3{1, 1, 1})
	joined := UnionSolid(s1, s2)
	expected := Dedent(`
		fn sym_fn_0_sphere_solid(p: vec3<f32>) -> bool {
			let center = vec3<f32>(0.0, 0.0, 0.0);
			return distance(p, center) <= 1.000000;
		}
		fn sym_fn_1_sphere_solid(p: vec3<f32>) -> bool {
			let center = vec3<f32>(0.0, 0.0, 0.0);
			return distance(p, center) <= 0.500000;
		}
		fn sym_fn_2_translate(p: vec3<f32>) -> bool {
			let newP = p - vec3<f32>(1.000000, 1.000000, 1.000000);
			return sym_fn_1_sphere_solid(newP);
		}
		fn sym_fn_3_union_solid(p: vec3<f32>) -> bool {
			return sym_fn_0_sphere_solid(p) || sym_fn_2_translate(p);
		}
	`)
	if joined.Code != expected {
		t.Fatalf("unexpected code: %s", joined.Code)
	}
}
