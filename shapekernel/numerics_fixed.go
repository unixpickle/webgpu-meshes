package shapekernel

import "math"

const fixed64Scale = 4294967296.0

var Fixed64Numerics Numerics = Numerics{
	Library: Dedent(`
		struct Fixed64 {
			whole: i32,
			frac: u32,
		}

		struct Fixed64v2 {
			x: Fixed64,
			y: Fixed64,
		}

		struct Fixed64v3 {
			x: Fixed64,
			y: Fixed64,
			z: Fixed64,
		}

		fn fixed64_new(whole: i32, frac: u32) -> Fixed64 {
			return Fixed64(whole, frac);
		}

		fn fixed64_as_float(x: Fixed64) -> f32 {
			return f32(x.whole) + f32(x.frac) / 4294967296.0;
		}

		fn fixed64_as_float2(x: Fixed64v2) -> vec2<f32> {
			return vec2<f32>(fixed64_as_float(x.x), fixed64_as_float(x.y));
		}

		fn fixed64_as_float3(x: Fixed64v3) -> vec3<f32> {
			return vec3<f32>(fixed64_as_float(x.x), fixed64_as_float(x.y), fixed64_as_float(x.z));
		}

		fn fixed64_from_float(x: f32) -> Fixed64 {
			if (x >= 2147483647.0) {
				return fixed64_pos_inf();
			}
			if (x <= -2147483648.0) {
				return fixed64_neg_inf();
			}
			let whole_f = floor(x);
			let frac_f = round((x - whole_f) * 4294967296.0);
			if (frac_f >= 4294967296.0) {
				return Fixed64(i32(whole_f) + 1, 0u);
			}
			return Fixed64(i32(whole_f), u32(frac_f));
		}

		fn fixed64_pos_inf() -> Fixed64 {
			return Fixed64(2147483647, 0xffffffffu);
		}

		fn fixed64_neg_inf() -> Fixed64 {
			return Fixed64(-2147483648, 0u);
		}

		fn fixed64_make2(x: Fixed64, y: Fixed64) -> Fixed64v2 {
			return Fixed64v2(x, y);
		}

		fn fixed64_make3(x: Fixed64, y: Fixed64, z: Fixed64) -> Fixed64v3 {
			return Fixed64v3(x, y, z);
		}

		fn fixed64_add(x: Fixed64, y: Fixed64) -> Fixed64 {
			let frac = x.frac + y.frac;
			let carry = select(0, 1, frac < x.frac);
			return Fixed64(x.whole + y.whole + carry, frac);
		}

		fn fixed64_sub(x: Fixed64, y: Fixed64) -> Fixed64 {
			let frac = x.frac - y.frac;
			let borrow = select(0, 1, x.frac < y.frac);
			return Fixed64(x.whole - y.whole - borrow, frac);
		}

		fn fixed64_neg(x: Fixed64) -> Fixed64 {
			return fixed64_sub(Fixed64(0, 0u), x);
		}

		fn fixed64_mul(x: Fixed64, y: Fixed64) -> Fixed64 {
			return fixed64_from_float(fixed64_as_float(x) * fixed64_as_float(y));
		}

		fn fixed64_div(x: Fixed64, y: Fixed64) -> Fixed64 {
			return fixed64_from_float(fixed64_as_float(x) / fixed64_as_float(y));
		}

		fn fixed64_add2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_add(x.x, y.x), fixed64_add(x.y, y.y));
		}

		fn fixed64_sub2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_sub(x.x, y.x), fixed64_sub(x.y, y.y));
		}

		fn fixed64_mul2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_mul(x.x, y.x), fixed64_mul(x.y, y.y));
		}

		fn fixed64_div2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_div(x.x, y.x), fixed64_div(x.y, y.y));
		}

		fn fixed64_add3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_add(x.x, y.x), fixed64_add(x.y, y.y), fixed64_add(x.z, y.z));
		}

		fn fixed64_sub3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_sub(x.x, y.x), fixed64_sub(x.y, y.y), fixed64_sub(x.z, y.z));
		}

		fn fixed64_mul3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_mul(x.x, y.x), fixed64_mul(x.y, y.y), fixed64_mul(x.z, y.z));
		}

		fn fixed64_div3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_div(x.x, y.x), fixed64_div(x.y, y.y), fixed64_div(x.z, y.z));
		}

		fn fixed64_get2(x: Fixed64v2, idx: i32) -> Fixed64 {
			if (idx == 0) {
				return x.x;
			}
			return x.y;
		}

		fn fixed64_get2_x(x: Fixed64v2) -> Fixed64 {
			return x.x;
		}

		fn fixed64_get2_y(x: Fixed64v2) -> Fixed64 {
			return x.y;
		}

		fn fixed64_get3(x: Fixed64v3, idx: i32) -> Fixed64 {
			if (idx == 0) {
				return x.x;
			}
			if (idx == 1) {
				return x.y;
			}
			return x.z;
		}

		fn fixed64_get3_x(x: Fixed64v3) -> Fixed64 {
			return x.x;
		}

		fn fixed64_get3_y(x: Fixed64v3) -> Fixed64 {
			return x.y;
		}

		fn fixed64_get3_z(x: Fixed64v3) -> Fixed64 {
			return x.z;
		}

		fn fixed64_eq(x: Fixed64, y: Fixed64) -> bool {
			return x.whole == y.whole && x.frac == y.frac;
		}

		fn fixed64_lt(x: Fixed64, y: Fixed64) -> bool {
			return x.whole < y.whole || (x.whole == y.whole && x.frac < y.frac);
		}

		fn fixed64_gt(x: Fixed64, y: Fixed64) -> bool {
			return fixed64_lt(y, x);
		}

		fn fixed64_le(x: Fixed64, y: Fixed64) -> bool {
			return !fixed64_gt(x, y);
		}

		fn fixed64_ge(x: Fixed64, y: Fixed64) -> bool {
			return !fixed64_lt(x, y);
		}

		fn fixed64_eq2(x: Fixed64v2, y: Fixed64v2) -> vec2<bool> {
			return vec2<bool>(fixed64_eq(x.x, y.x), fixed64_eq(x.y, y.y));
		}

		fn fixed64_lt2(x: Fixed64v2, y: Fixed64v2) -> vec2<bool> {
			return vec2<bool>(fixed64_lt(x.x, y.x), fixed64_lt(x.y, y.y));
		}

		fn fixed64_gt2(x: Fixed64v2, y: Fixed64v2) -> vec2<bool> {
			return vec2<bool>(fixed64_gt(x.x, y.x), fixed64_gt(x.y, y.y));
		}

		fn fixed64_le2(x: Fixed64v2, y: Fixed64v2) -> vec2<bool> {
			return vec2<bool>(fixed64_le(x.x, y.x), fixed64_le(x.y, y.y));
		}

		fn fixed64_ge2(x: Fixed64v2, y: Fixed64v2) -> vec2<bool> {
			return vec2<bool>(fixed64_ge(x.x, y.x), fixed64_ge(x.y, y.y));
		}

		fn fixed64_eq3(x: Fixed64v3, y: Fixed64v3) -> vec3<bool> {
			return vec3<bool>(fixed64_eq(x.x, y.x), fixed64_eq(x.y, y.y), fixed64_eq(x.z, y.z));
		}

		fn fixed64_lt3(x: Fixed64v3, y: Fixed64v3) -> vec3<bool> {
			return vec3<bool>(fixed64_lt(x.x, y.x), fixed64_lt(x.y, y.y), fixed64_lt(x.z, y.z));
		}

		fn fixed64_gt3(x: Fixed64v3, y: Fixed64v3) -> vec3<bool> {
			return vec3<bool>(fixed64_gt(x.x, y.x), fixed64_gt(x.y, y.y), fixed64_gt(x.z, y.z));
		}

		fn fixed64_le3(x: Fixed64v3, y: Fixed64v3) -> vec3<bool> {
			return vec3<bool>(fixed64_le(x.x, y.x), fixed64_le(x.y, y.y), fixed64_le(x.z, y.z));
		}

		fn fixed64_ge3(x: Fixed64v3, y: Fixed64v3) -> vec3<bool> {
			return vec3<bool>(fixed64_ge(x.x, y.x), fixed64_ge(x.y, y.y), fixed64_ge(x.z, y.z));
		}

		fn fixed64_abs(x: Fixed64) -> Fixed64 {
			if (fixed64_lt(x, Fixed64(0, 0u))) {
				return fixed64_neg(x);
			}
			return x;
		}

		fn fixed64_sqrt(x: Fixed64) -> Fixed64 {
			return fixed64_from_float(sqrt(fixed64_as_float(x)));
		}

		fn fixed64_min(x: Fixed64, y: Fixed64) -> Fixed64 {
			if (fixed64_lt(x, y)) {
				return x;
			}
			return y;
		}

		fn fixed64_max(x: Fixed64, y: Fixed64) -> Fixed64 {
			if (fixed64_gt(x, y)) {
				return x;
			}
			return y;
		}

		fn fixed64_pow(x: Fixed64, y: Fixed64) -> Fixed64 {
			return fixed64_from_float(pow(fixed64_as_float(x), fixed64_as_float(y)));
		}

		fn fixed64_exp(x: Fixed64) -> Fixed64 {
			return fixed64_from_float(exp(fixed64_as_float(x)));
		}

		fn fixed64_cos(x: Fixed64) -> Fixed64 {
			return fixed64_from_float(cos(fixed64_as_float(x)));
		}

		fn fixed64_sin(x: Fixed64) -> Fixed64 {
			return fixed64_from_float(sin(fixed64_as_float(x)));
		}

		fn fixed64_clamp(x: Fixed64, mi: Fixed64, ma: Fixed64) -> Fixed64 {
			return fixed64_min(fixed64_max(x, mi), ma);
		}

		fn fixed64_abs2(x: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_abs(x.x), fixed64_abs(x.y));
		}

		fn fixed64_sqrt2(x: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_sqrt(x.x), fixed64_sqrt(x.y));
		}

		fn fixed64_min2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_min(x.x, y.x), fixed64_min(x.y, y.y));
		}

		fn fixed64_max2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_max(x.x, y.x), fixed64_max(x.y, y.y));
		}

		fn fixed64_pow2(x: Fixed64v2, y: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_pow(x.x, y.x), fixed64_pow(x.y, y.y));
		}

		fn fixed64_exp2(x: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_exp(x.x), fixed64_exp(x.y));
		}

		fn fixed64_cos2(x: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_cos(x.x), fixed64_cos(x.y));
		}

		fn fixed64_sin2(x: Fixed64v2) -> Fixed64v2 {
			return Fixed64v2(fixed64_sin(x.x), fixed64_sin(x.y));
		}

		fn fixed64_abs3(x: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_abs(x.x), fixed64_abs(x.y), fixed64_abs(x.z));
		}

		fn fixed64_sqrt3(x: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_sqrt(x.x), fixed64_sqrt(x.y), fixed64_sqrt(x.z));
		}

		fn fixed64_min3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_min(x.x, y.x), fixed64_min(x.y, y.y), fixed64_min(x.z, y.z));
		}

		fn fixed64_max3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_max(x.x, y.x), fixed64_max(x.y, y.y), fixed64_max(x.z, y.z));
		}

		fn fixed64_pow3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_pow(x.x, y.x), fixed64_pow(x.y, y.y), fixed64_pow(x.z, y.z));
		}

		fn fixed64_exp3(x: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_exp(x.x), fixed64_exp(x.y), fixed64_exp(x.z));
		}

		fn fixed64_cos3(x: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_cos(x.x), fixed64_cos(x.y), fixed64_cos(x.z));
		}

		fn fixed64_sin3(x: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(fixed64_sin(x.x), fixed64_sin(x.y), fixed64_sin(x.z));
		}

		fn fixed64_dot2(x: Fixed64v2, y: Fixed64v2) -> Fixed64 {
			return fixed64_add(fixed64_mul(x.x, y.x), fixed64_mul(x.y, y.y));
		}

		fn fixed64_dot3(x: Fixed64v3, y: Fixed64v3) -> Fixed64 {
			return fixed64_add(fixed64_add(fixed64_mul(x.x, y.x), fixed64_mul(x.y, y.y)), fixed64_mul(x.z, y.z));
		}

		fn fixed64_len2(x: Fixed64v2) -> Fixed64 {
			return fixed64_sqrt(fixed64_dot2(x, x));
		}

		fn fixed64_len3(x: Fixed64v3) -> Fixed64 {
			return fixed64_sqrt(fixed64_dot3(x, x));
		}

		fn fixed64_dist2(x: Fixed64v2, y: Fixed64v2) -> Fixed64 {
			return fixed64_len2(fixed64_sub2(x, y));
		}

		fn fixed64_dist3(x: Fixed64v3, y: Fixed64v3) -> Fixed64 {
			return fixed64_len3(fixed64_sub3(x, y));
		}

		fn fixed64_cross3(x: Fixed64v3, y: Fixed64v3) -> Fixed64v3 {
			return Fixed64v3(
				fixed64_sub(fixed64_mul(x.y, y.z), fixed64_mul(x.z, y.y)),
				fixed64_sub(fixed64_mul(x.z, y.x), fixed64_mul(x.x, y.z)),
				fixed64_sub(fixed64_mul(x.x, y.y), fixed64_mul(x.y, y.x)),
			);
		}

		fn fixed64_scale2(x: Fixed64v2, y: Fixed64) -> Fixed64v2 {
			return Fixed64v2(fixed64_mul(x.x, y), fixed64_mul(x.y, y));
		}

		fn fixed64_scale3(x: Fixed64v3, y: Fixed64) -> Fixed64v3 {
			return Fixed64v3(fixed64_mul(x.x, y), fixed64_mul(x.y, y), fixed64_mul(x.z, y));
		}
	`),
	Symbols: NumericSymbols{
		Dtype:     "Fixed64",
		Dtype2:    "Fixed64v2",
		Dtype3:    "Fixed64v3",
		AsFloat:   "fixed64_as_float",
		AsFloat2:  "fixed64_as_float2",
		AsFloat3:  "fixed64_as_float3",
		FromFloat: "fixed64_from_float",
		Make2:     "fixed64_make2",
		Make3:     "fixed64_make3",
		Add:       "fixed64_add",
		Sub:       "fixed64_sub",
		Div:       "fixed64_div",
		Mul:       "fixed64_mul",
		Add2:      "fixed64_add2",
		Sub2:      "fixed64_sub2",
		Div2:      "fixed64_div2",
		Mul2:      "fixed64_mul2",
		Add3:      "fixed64_add3",
		Sub3:      "fixed64_sub3",
		Div3:      "fixed64_div3",
		Mul3:      "fixed64_mul3",
		Get2:      "fixed64_get2",
		Get3:      "fixed64_get3",
		Get2X:     "fixed64_get2_x",
		Get2Y:     "fixed64_get2_y",
		Get3X:     "fixed64_get3_x",
		Get3Y:     "fixed64_get3_y",
		Get3Z:     "fixed64_get3_z",
		Lt:        "fixed64_lt",
		Eq:        "fixed64_eq",
		Gt:        "fixed64_gt",
		Le:        "fixed64_le",
		Ge:        "fixed64_ge",
		Lt2:       "fixed64_lt2",
		Eq2:       "fixed64_eq2",
		Gt2:       "fixed64_gt2",
		Le2:       "fixed64_le2",
		Ge2:       "fixed64_ge2",
		Lt3:       "fixed64_lt3",
		Eq3:       "fixed64_eq3",
		Gt3:       "fixed64_gt3",
		Le3:       "fixed64_le3",
		Ge3:       "fixed64_ge3",
		Sqrt:      "fixed64_sqrt",
		Sqrt2:     "fixed64_sqrt2",
		Sqrt3:     "fixed64_sqrt3",
		Abs:       "fixed64_abs",
		Abs2:      "fixed64_abs2",
		Abs3:      "fixed64_abs3",
		Min:       "fixed64_min",
		Min2:      "fixed64_min2",
		Min3:      "fixed64_min3",
		Max:       "fixed64_max",
		Max2:      "fixed64_max2",
		Max3:      "fixed64_max3",
		Pow:       "fixed64_pow",
		Pow2:      "fixed64_pow2",
		Pow3:      "fixed64_pow3",
		Exp:       "fixed64_exp",
		Exp2:      "fixed64_exp2",
		Exp3:      "fixed64_exp3",
		Cos:       "fixed64_cos",
		Cos2:      "fixed64_cos2",
		Cos3:      "fixed64_cos3",
		Sin:       "fixed64_sin",
		Sin2:      "fixed64_sin2",
		Sin3:      "fixed64_sin3",
		Clamp:     "fixed64_clamp",
		Dot2:      "fixed64_dot2",
		Dot3:      "fixed64_dot3",
		Dist2:     "fixed64_dist2",
		Dist3:     "fixed64_dist3",
		Len2:      "fixed64_len2",
		Len3:      "fixed64_len3",
		Cross3:    "fixed64_cross3",
		Scale2:    "fixed64_scale2",
		Scale3:    "fixed64_scale3",
		Zero:      "Fixed64(0, 0u)",
		One:       "Fixed64(1, 0u)",
	},
	Literal:  fixed64Literal,
	Infinity: fixed64Infinity,
}

func fixed64Literal(f float64) string {
	if math.IsInf(f, 1) {
		return fixed64Infinity(1)
	}
	if math.IsInf(f, -1) {
		return fixed64Infinity(-1)
	}
	if math.IsNaN(f) {
		return "Fixed64(0, 0u)"
	}

	maxValue := math.Nextafter(math.Pow(2, 31), 0)
	if f >= maxValue {
		return fixed64Infinity(1)
	}
	if f <= -math.Pow(2, 31) {
		return fixed64Infinity(-1)
	}

	scaled := math.Round(f * fixed64Scale)
	if scaled > math.MaxInt64 {
		scaled = math.MaxInt64
	} else if scaled < math.MinInt64 {
		scaled = math.MinInt64
	}
	value := int64(scaled)
	return fixed64LiteralBits(value)
}

func fixed64LiteralBits(value int64) string {
	return WGSL(
		"Fixed64({{.Whole}}, {{.Frac}}u)",
		"Whole", int32(value>>32),
		"Frac", uint32(value),
	)
}

func fixed64Infinity(sign int) string {
	if sign < 0 {
		return "fixed64_neg_inf()"
	}
	return "fixed64_pos_inf()"
}
