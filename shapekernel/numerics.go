package shapekernel

type NumericSymbols struct {
	Dtype  string
	Dtype2 string
	Dtype3 string

	AsFloat   string
	AsFloat2  string
	AsFloat3  string
	FromFloat string
	Make2     string
	Make3     string

	Add  string
	Sub  string
	Div  string
	Mul  string
	Add2 string
	Sub2 string
	Div2 string
	Mul2 string
	Add3 string
	Sub3 string
	Div3 string
	Mul3 string

	Get2  string
	Get3  string
	Get2X string
	Get2Y string
	Get3X string
	Get3Y string
	Get3Z string

	Lt  string
	Eq  string
	Gt  string
	Le  string
	Ge  string
	Lt2 string
	Eq2 string
	Gt2 string
	Le2 string
	Ge2 string
	Lt3 string
	Eq3 string
	Gt3 string
	Le3 string
	Ge3 string

	// Mathematical functions, elemwise for vectors.
	Sqrt  string
	Sqrt2 string
	Sqrt3 string
	Abs   string
	Abs2  string
	Abs3  string
	Min   string
	Min2  string
	Min3  string
	Max   string
	Max2  string
	Max3  string
	Pow   string
	Pow2  string
	Pow3  string
	Exp   string
	Exp2  string
	Exp3  string
	Cos   string
	Cos2  string
	Cos3  string
	Sin   string
	Sin2  string
	Sin3  string
	Clamp string

	// Vector operations
	Dot2   string
	Dot3   string
	Dist2  string
	Dist3  string
	Len2   string
	Len3   string
	Cross3 string
	Scale2 string
	Scale3 string

	// Constants
	Zero string
	One  string
}

type Numerics struct {
	Library  string
	Symbols  NumericSymbols
	Literal  func(f float64) string
	Infinity func(sign int) string
}

var NativeFloat32Numerics Numerics = Numerics{
	Library: Dedent(`
		fn num_f32_as_float(x: f32) -> f32 {
			return x;
		}

		fn num_f32_pos_inf() -> f32 {
			var zero = 0.0;
			return 1.0 / zero;
		}

		fn num_f32_neg_inf() -> f32 {
			var zero = 0.0;
			return -1.0 / zero;
		}

		fn num_f32_as_float2(x: vec2<f32>) -> vec2<f32> {
			return x;
		}

		fn num_f32_as_float3(x: vec3<f32>) -> vec3<f32> {
			return x;
		}

		fn num_f32_from_float(x: f32) -> f32 {
			return x;
		}

		fn num_f32_make2(x: f32, y: f32) -> vec2<f32> {
			return vec2<f32>(x, y);
		}

		fn num_f32_make3(x: f32, y: f32, z: f32) -> vec3<f32> {
			return vec3<f32>(x, y, z);
		}
	
		fn num_f32_add(x: f32, y: f32) -> f32 {
			return x + y;
		}

		fn num_f32_add2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return x + y;
		}

		fn num_f32_add3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return x + y;
		}

		fn num_f32_sub(x: f32, y: f32) -> f32 {
			return x - y;
		}

		fn num_f32_sub2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return x - y;
		}

		fn num_f32_sub3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return x - y;
		}

		fn num_f32_div(x: f32, y: f32) -> f32 {
			return x / y;
		}

		fn num_f32_div2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return x / y;
		}

		fn num_f32_div3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return x / y;
		}

		fn num_f32_mul(x: f32, y: f32) -> f32 {
			return x * y;
		}

		fn num_f32_mul2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return x * y;
		}

		fn num_f32_mul3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return x * y;
		}

		fn num_f32_get2(x: vec2<f32>, idx: i32) -> f32 {
			return x[idx];
		}

		fn num_f32_get2_x(x: vec2<f32>) -> f32 {
			return x.x;
		}

		fn num_f32_get2_y(x: vec2<f32>) -> f32 {
			return x.y;
		}

		fn num_f32_get3(x: vec3<f32>, idx: i32) -> f32 {
			return x[idx];
		}

		fn num_f32_get3_x(x: vec3<f32>) -> f32 {
			return x.x;
		}

		fn num_f32_get3_y(x: vec3<f32>) -> f32 {
			return x.y;
		}

		fn num_f32_get3_z(x: vec3<f32>) -> f32 {
			return x.z;
		}

		fn num_f32_eq(x: f32, y: f32) -> bool {
			return x == y;
		}

		fn num_f32_lt(x: f32, y: f32) -> bool {
			return x < y;
		}

		fn num_f32_gt(x: f32, y: f32) -> bool {
			return x > y;
		}

		fn num_f32_le(x: f32, y: f32) -> bool {
			return x <= y;
		}

		fn num_f32_ge(x: f32, y: f32) -> bool {
			return x >= y;
		}

		fn num_f32_lt2(x: vec2<f32>, y: vec2<f32>) -> vec2<bool> {
			return x < y;
		}

		fn num_f32_eq2(x: vec2<f32>, y: vec2<f32>) -> vec2<bool> {
			return x == y;
		}

		fn num_f32_gt2(x: vec2<f32>, y: vec2<f32>) -> vec2<bool> {
			return x > y;
		}

		fn num_f32_le2(x: vec2<f32>, y: vec2<f32>) -> vec2<bool> {
			return x <= y;
		}

		fn num_f32_ge2(x: vec2<f32>, y: vec2<f32>) -> vec2<bool> {
			return x >= y;
		}

		fn num_f32_lt3(x: vec3<f32>, y: vec3<f32>) -> vec3<bool> {
			return x < y;
		}

		fn num_f32_eq3(x: vec3<f32>, y: vec3<f32>) -> vec3<bool> {
			return x == y;
		}

		fn num_f32_gt3(x: vec3<f32>, y: vec3<f32>) -> vec3<bool> {
			return x > y;
		}

		fn num_f32_le3(x: vec3<f32>, y: vec3<f32>) -> vec3<bool> {
			return x <= y;
		}

		fn num_f32_ge3(x: vec3<f32>, y: vec3<f32>) -> vec3<bool> {
			return x >= y;
		}

		fn num_f32_sqrt(x: f32) -> f32 {
			return sqrt(x);
		}

		fn num_f32_sqrt2(x: vec2<f32>) -> vec2<f32> {
			return sqrt(x);
		}

		fn num_f32_sqrt3(x: vec3<f32>) -> vec3<f32> {
			return sqrt(x);
		}

		fn num_f32_abs(x: f32) -> f32 {
			return abs(x);
		}

		fn num_f32_abs2(x: vec2<f32>) -> vec2<f32> {
			return abs(x);
		}

		fn num_f32_abs3(x: vec3<f32>) -> vec3<f32> {
			return abs(x);
		}

		fn num_f32_min(x: f32, y: f32) -> f32 {
			return min(x, y);
		}

		fn num_f32_min2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return min(x, y);
		}

		fn num_f32_min3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return min(x, y);
		}

		fn num_f32_max(x: f32, y: f32) -> f32 {
			return max(x, y);
		}

		fn num_f32_max2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return max(x, y);
		}

		fn num_f32_max3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return max(x, y);
		}

		fn num_f32_pow(x: f32, y: f32) -> f32 {
			return pow(x, y);
		}

		fn num_f32_pow2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
			return pow(x, y);
		}

		fn num_f32_pow3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return pow(x, y);
		}

		fn num_f32_exp(x: f32) -> f32 {
			return exp(x);
		}

		fn num_f32_exp2(x: vec2<f32>) -> vec2<f32> {
			return exp(x);
		}

		fn num_f32_exp3(x: vec3<f32>) -> vec3<f32> {
			return exp(x);
		}

		fn num_f32_cos(x: f32) -> f32 {
			return cos(x);
		}

		fn num_f32_cos2(x: vec2<f32>) -> vec2<f32> {
			return cos(x);
		}

		fn num_f32_cos3(x: vec3<f32>) -> vec3<f32> {
			return cos(x);
		}

		fn num_f32_sin(x: f32) -> f32 {
			return sin(x);
		}

		fn num_f32_sin2(x: vec2<f32>) -> vec2<f32> {
			return sin(x);
		}

		fn num_f32_sin3(x: vec3<f32>) -> vec3<f32> {
			return sin(x);
		}

		fn num_f32_clamp(x: f32, mi: f32, ma: f32) -> f32 {
			return clamp(x, mi, ma);
		}

		fn num_f32_dot2(x: vec2<f32>, y: vec2<f32>) -> f32 {
			return dot(x, y);
		}

		fn num_f32_dot3(x: vec3<f32>, y: vec3<f32>) -> f32 {
			return dot(x, y);
		}

		fn num_f32_dist2(x: vec2<f32>, y: vec2<f32>) -> f32 {
			return distance(x, y);
		}

		fn num_f32_dist3(x: vec3<f32>, y: vec3<f32>) -> f32 {
			return distance(x, y);
		}

		fn num_f32_len2(x: vec2<f32>) -> f32 {
			return length(x);
		}

		fn num_f32_len3(x: vec3<f32>) -> f32 {
			return length(x);
		}

		fn num_f32_cross3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
			return cross(x, y);
		}

		fn num_f32_scale2(x: vec2<f32>, y: f32) -> vec2<f32> {
			return x * y;
		}

		fn num_f32_scale3(x: vec3<f32>, y: f32) -> vec3<f32> {
			return x * y;
		}
	`),
	Symbols: NumericSymbols{
		Dtype:     "f32",
		Dtype2:    "vec2<f32>",
		Dtype3:    "vec3<f32>",
		AsFloat:   "num_f32_as_float",
		AsFloat2:  "num_f32_as_float2",
		AsFloat3:  "num_f32_as_float3",
		FromFloat: "num_f32_from_float",
		Make2:     "num_f32_make2",
		Make3:     "num_f32_make3",
		Add:       "num_f32_add",
		Sub:       "num_f32_sub",
		Div:       "num_f32_div",
		Mul:       "num_f32_mul",
		Add2:      "num_f32_add2",
		Sub2:      "num_f32_sub2",
		Div2:      "num_f32_div2",
		Mul2:      "num_f32_mul2",
		Add3:      "num_f32_add3",
		Sub3:      "num_f32_sub3",
		Div3:      "num_f32_div3",
		Mul3:      "num_f32_mul3",
		Get2:      "num_f32_get2",
		Get3:      "num_f32_get3",
		Get2X:     "num_f32_get2_x",
		Get2Y:     "num_f32_get2_y",
		Get3X:     "num_f32_get3_x",
		Get3Y:     "num_f32_get3_y",
		Get3Z:     "num_f32_get3_z",
		Lt:        "num_f32_lt",
		Eq:        "num_f32_eq",
		Gt:        "num_f32_gt",
		Le:        "num_f32_le",
		Ge:        "num_f32_ge",
		Lt2:       "num_f32_lt2",
		Eq2:       "num_f32_eq2",
		Gt2:       "num_f32_gt2",
		Le2:       "num_f32_le2",
		Ge2:       "num_f32_ge2",
		Lt3:       "num_f32_lt3",
		Eq3:       "num_f32_eq3",
		Gt3:       "num_f32_gt3",
		Le3:       "num_f32_le3",
		Ge3:       "num_f32_ge3",
		Sqrt:      "num_f32_sqrt",
		Sqrt2:     "num_f32_sqrt2",
		Sqrt3:     "num_f32_sqrt3",
		Abs:       "num_f32_abs",
		Abs2:      "num_f32_abs2",
		Abs3:      "num_f32_abs3",
		Min:       "num_f32_min",
		Min2:      "num_f32_min2",
		Min3:      "num_f32_min3",
		Max:       "num_f32_max",
		Max2:      "num_f32_max2",
		Max3:      "num_f32_max3",
		Pow:       "num_f32_pow",
		Pow2:      "num_f32_pow2",
		Pow3:      "num_f32_pow3",
		Exp:       "num_f32_exp",
		Exp2:      "num_f32_exp2",
		Exp3:      "num_f32_exp3",
		Cos:       "num_f32_cos",
		Cos2:      "num_f32_cos2",
		Cos3:      "num_f32_cos3",
		Sin:       "num_f32_sin",
		Sin2:      "num_f32_sin2",
		Sin3:      "num_f32_sin3",
		Clamp:     "num_f32_clamp",
		Dot2:      "num_f32_dot2",
		Dot3:      "num_f32_dot3",
		Dist2:     "num_f32_dist2",
		Dist3:     "num_f32_dist3",
		Len2:      "num_f32_len2",
		Len3:      "num_f32_len3",
		Cross3:    "num_f32_cross3",
		Scale2:    "num_f32_scale2",
		Scale3:    "num_f32_scale3",

		Zero: "0.0",
		One:  "1.0",
	},
	Literal: func(f float64) string {
		return wgslFloatLiteral(f, 32)
	},
	Infinity: func(sign int) string {
		if sign < 0 {
			return "num_f32_neg_inf()"
		}
		return "num_f32_pos_inf()"
	},
}
