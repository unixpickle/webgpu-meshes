package shapekernel

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
)

var SmokeFloat32Numerics Numerics = Numerics{
	Library: Dedent(`
		struct SmokeF32 {
			value: f32,
		}

		struct SmokeF32v2 {
			value: vec2<f32>,
		}

		struct SmokeF32v3 {
			value: vec3<f32>,
		}

		fn smoke_f32_as_float(x: SmokeF32) -> f32 {
			return x.value;
		}

		fn smoke_f32_pos_inf() -> SmokeF32 {
			var zero = 0.0;
			return SmokeF32(1.0 / zero);
		}

		fn smoke_f32_neg_inf() -> SmokeF32 {
			var zero = 0.0;
			return SmokeF32(-1.0 / zero);
		}

		fn smoke_f32_as_float2(x: SmokeF32v2) -> vec2<f32> {
			return x.value;
		}

		fn smoke_f32_as_float3(x: SmokeF32v3) -> vec3<f32> {
			return x.value;
		}

		fn smoke_f32_from_float(x: f32) -> SmokeF32 {
			return SmokeF32(x);
		}

		fn smoke_f32_make2(x: SmokeF32, y: SmokeF32) -> SmokeF32v2 {
			return SmokeF32v2(vec2<f32>(x.value, y.value));
		}

		fn smoke_f32_make3(x: SmokeF32, y: SmokeF32, z: SmokeF32) -> SmokeF32v3 {
			return SmokeF32v3(vec3<f32>(x.value, y.value, z.value));
		}

		fn smoke_f32_add(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(x.value + y.value);
		}

		fn smoke_f32_add2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(x.value + y.value);
		}

		fn smoke_f32_add3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(x.value + y.value);
		}

		fn smoke_f32_sub(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(x.value - y.value);
		}

		fn smoke_f32_sub2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(x.value - y.value);
		}

		fn smoke_f32_sub3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(x.value - y.value);
		}

		fn smoke_f32_div(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(x.value / y.value);
		}

		fn smoke_f32_div2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(x.value / y.value);
		}

		fn smoke_f32_div3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(x.value / y.value);
		}

		fn smoke_f32_mul(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(x.value * y.value);
		}

		fn smoke_f32_mul2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(x.value * y.value);
		}

		fn smoke_f32_mul3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(x.value * y.value);
		}

		fn smoke_f32_get2(x: SmokeF32v2, idx: i32) -> SmokeF32 {
			return SmokeF32(x.value[idx]);
		}

		fn smoke_f32_get2_x(x: SmokeF32v2) -> SmokeF32 {
			return SmokeF32(x.value.x);
		}

		fn smoke_f32_get2_y(x: SmokeF32v2) -> SmokeF32 {
			return SmokeF32(x.value.y);
		}

		fn smoke_f32_get3(x: SmokeF32v3, idx: i32) -> SmokeF32 {
			return SmokeF32(x.value[idx]);
		}

		fn smoke_f32_get3_x(x: SmokeF32v3) -> SmokeF32 {
			return SmokeF32(x.value.x);
		}

		fn smoke_f32_get3_y(x: SmokeF32v3) -> SmokeF32 {
			return SmokeF32(x.value.y);
		}

		fn smoke_f32_get3_z(x: SmokeF32v3) -> SmokeF32 {
			return SmokeF32(x.value.z);
		}

		fn smoke_f32_eq(x: SmokeF32, y: SmokeF32) -> bool {
			return x.value == y.value;
		}

		fn smoke_f32_lt(x: SmokeF32, y: SmokeF32) -> bool {
			return x.value < y.value;
		}

		fn smoke_f32_gt(x: SmokeF32, y: SmokeF32) -> bool {
			return x.value > y.value;
		}

		fn smoke_f32_le(x: SmokeF32, y: SmokeF32) -> bool {
			return x.value <= y.value;
		}

		fn smoke_f32_ge(x: SmokeF32, y: SmokeF32) -> bool {
			return x.value >= y.value;
		}

		fn smoke_f32_lt2(x: SmokeF32v2, y: SmokeF32v2) -> vec2<bool> {
			return x.value < y.value;
		}

		fn smoke_f32_eq2(x: SmokeF32v2, y: SmokeF32v2) -> vec2<bool> {
			return x.value == y.value;
		}

		fn smoke_f32_gt2(x: SmokeF32v2, y: SmokeF32v2) -> vec2<bool> {
			return x.value > y.value;
		}

		fn smoke_f32_le2(x: SmokeF32v2, y: SmokeF32v2) -> vec2<bool> {
			return x.value <= y.value;
		}

		fn smoke_f32_ge2(x: SmokeF32v2, y: SmokeF32v2) -> vec2<bool> {
			return x.value >= y.value;
		}

		fn smoke_f32_lt3(x: SmokeF32v3, y: SmokeF32v3) -> vec3<bool> {
			return x.value < y.value;
		}

		fn smoke_f32_eq3(x: SmokeF32v3, y: SmokeF32v3) -> vec3<bool> {
			return x.value == y.value;
		}

		fn smoke_f32_gt3(x: SmokeF32v3, y: SmokeF32v3) -> vec3<bool> {
			return x.value > y.value;
		}

		fn smoke_f32_le3(x: SmokeF32v3, y: SmokeF32v3) -> vec3<bool> {
			return x.value <= y.value;
		}

		fn smoke_f32_ge3(x: SmokeF32v3, y: SmokeF32v3) -> vec3<bool> {
			return x.value >= y.value;
		}

		fn smoke_f32_sqrt(x: SmokeF32) -> SmokeF32 {
			return SmokeF32(sqrt(x.value));
		}

		fn smoke_f32_sqrt2(x: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(sqrt(x.value));
		}

		fn smoke_f32_sqrt3(x: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(sqrt(x.value));
		}

		fn smoke_f32_abs(x: SmokeF32) -> SmokeF32 {
			return SmokeF32(abs(x.value));
		}

		fn smoke_f32_abs2(x: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(abs(x.value));
		}

		fn smoke_f32_abs3(x: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(abs(x.value));
		}

		fn smoke_f32_min(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(min(x.value, y.value));
		}

		fn smoke_f32_min2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(min(x.value, y.value));
		}

		fn smoke_f32_min3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(min(x.value, y.value));
		}

		fn smoke_f32_max(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(max(x.value, y.value));
		}

		fn smoke_f32_max2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(max(x.value, y.value));
		}

		fn smoke_f32_max3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(max(x.value, y.value));
		}

		fn smoke_f32_pow(x: SmokeF32, y: SmokeF32) -> SmokeF32 {
			return SmokeF32(pow(x.value, y.value));
		}

		fn smoke_f32_pow2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(pow(x.value, y.value));
		}

		fn smoke_f32_pow3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(pow(x.value, y.value));
		}

		fn smoke_f32_exp(x: SmokeF32) -> SmokeF32 {
			return SmokeF32(exp(x.value));
		}

		fn smoke_f32_exp2(x: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(exp(x.value));
		}

		fn smoke_f32_exp3(x: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(exp(x.value));
		}

		fn smoke_f32_cos(x: SmokeF32) -> SmokeF32 {
			return SmokeF32(cos(x.value));
		}

		fn smoke_f32_cos2(x: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(cos(x.value));
		}

		fn smoke_f32_cos3(x: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(cos(x.value));
		}

		fn smoke_f32_sin(x: SmokeF32) -> SmokeF32 {
			return SmokeF32(sin(x.value));
		}

		fn smoke_f32_sin2(x: SmokeF32v2) -> SmokeF32v2 {
			return SmokeF32v2(sin(x.value));
		}

		fn smoke_f32_sin3(x: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(sin(x.value));
		}

		fn smoke_f32_clamp(x: SmokeF32, mi: SmokeF32, ma: SmokeF32) -> SmokeF32 {
			return SmokeF32(clamp(x.value, mi.value, ma.value));
		}

		fn smoke_f32_dot2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32 {
			return SmokeF32(dot(x.value, y.value));
		}

		fn smoke_f32_dot3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32 {
			return SmokeF32(dot(x.value, y.value));
		}

		fn smoke_f32_dist2(x: SmokeF32v2, y: SmokeF32v2) -> SmokeF32 {
			return SmokeF32(distance(x.value, y.value));
		}

		fn smoke_f32_dist3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32 {
			return SmokeF32(distance(x.value, y.value));
		}

		fn smoke_f32_len2(x: SmokeF32v2) -> SmokeF32 {
			return SmokeF32(length(x.value));
		}

		fn smoke_f32_len3(x: SmokeF32v3) -> SmokeF32 {
			return SmokeF32(length(x.value));
		}

		fn smoke_f32_cross3(x: SmokeF32v3, y: SmokeF32v3) -> SmokeF32v3 {
			return SmokeF32v3(cross(x.value, y.value));
		}

		fn smoke_f32_scale2(x: SmokeF32v2, y: SmokeF32) -> SmokeF32v2 {
			return SmokeF32v2(x.value * y.value);
		}

		fn smoke_f32_scale3(x: SmokeF32v3, y: SmokeF32) -> SmokeF32v3 {
			return SmokeF32v3(x.value * y.value);
		}
	`),
	Symbols: func() NumericSymbols {
		s := NativeFloat32Numerics.Symbols
		values := reflect.ValueOf(&s).Elem()
		for i := 0; i < values.NumField(); i++ {
			field := values.Type().Field(i)
			switch field.Name {
			case "Dtype", "Dtype2", "Dtype3", "Zero", "One":
				continue
			}
			values.Field(i).SetString(strings.ReplaceAll(values.Field(i).String(), "num_f32_", "smoke_f32_"))
		}
		s.Dtype = "SmokeF32"
		s.Dtype2 = "SmokeF32v2"
		s.Dtype3 = "SmokeF32v3"
		s.Zero = "SmokeF32(0.0)"
		s.One = "SmokeF32(1.0)"
		return s
	}(),
	Literal: func(f float64) string {
		return "SmokeF32(" + wgslFloatLiteral(f, 32) + ")"
	},
	Infinity: func(sign int) string {
		if sign < 0 {
			return "smoke_f32_neg_inf()"
		}
		return "smoke_f32_pos_inf()"
	},
}

func TestNativeFloat32NumericsSymbols(t *testing.T) {
	testNumericsSymbols(t, "NativeFloat32Numerics", NativeFloat32Numerics)
}

func TestSmokeFloat32NumericsSymbols(t *testing.T) {
	testNumericsSymbols(t, "SmokeFloat32Numerics", SmokeFloat32Numerics)
}

func testNumericsSymbols(t *testing.T, name string, n Numerics) {
	t.Helper()
	symbols := n.Symbols
	values := reflect.ValueOf(symbols)
	fields := values.Type()
	for i := 0; i < values.NumField(); i++ {
		field := fields.Field(i)
		value := values.Field(i).String()
		if value == "" {
			t.Fatalf("empty %s symbol: %s", name, field.Name)
		}
		switch field.Name {
		case "Dtype", "Dtype2", "Dtype3", "Zero", "One":
			continue
		}
		pattern := regexp.MustCompile(`fn\s+` + regexp.QuoteMeta(value) + `\s*\(`)
		if !pattern.MatchString(n.Library) {
			t.Fatalf("missing function definition for symbol %s=%q", field.Name, value)
		}
	}
}

func TestNativeFloat32NumericsLiteral(t *testing.T) {
	if got := NativeFloat32Numerics.Literal(float64(float32(1.0 / 3.0))); got != "0.33333334" {
		t.Fatalf("expected f32 literal precision, got %q", got)
	}
}

func TestSmokeFloat32NumericsLiteral(t *testing.T) {
	if got := SmokeFloat32Numerics.Literal(float64(float32(1.0 / 3.0))); got != "SmokeF32(0.33333334)" {
		t.Fatalf("expected wrapped f32 literal precision, got %q", got)
	}
}
