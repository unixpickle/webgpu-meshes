package shapekernel

import (
	"math"
	"testing"
)

func TestFixed64NumericsSymbols(t *testing.T) {
	testNumericsSymbols(t, "Fixed64Numerics", Fixed64Numerics)
}

func TestFixed64NumericsScalarOps(t *testing.T) {
	k := testFixed64Kernel(
		SDF2D,
		"scalar_ops",
		`
			let x = {{.N.Get2X}}(p);
			let y = {{.N.Get2Y}}(p);
			let sum = {{.N.Add}}(x, y);
			let diff = {{.N.Sub}}(x, y);
			let scaled_sum = {{.N.Mul}}(sum, {{.Half}});
			let scaled_diff = {{.N.Div}}(diff, {{.Four}});
			return {{.N.Sub}}(scaled_sum, scaled_diff);
		`,
	)
	vals := ExecuteShapeKernel(
		t,
		kernelToNative(Fixed64Numerics, k),
		Vec2{1.25, -0.5},
		Vec2{-2.5, 1.0},
		Vec2{0.375, 0.125},
	)
	vals.ExpectFloats(t, []float32{-0.0625, 0.125, 0.1875}, 1e-4)
}

func TestFixed64NumericsMathOps(t *testing.T) {
	k := testFixed64Kernel(
		SDF2D,
		"math_ops",
		`
			let x = {{.N.Get2X}}(p);
			let y = {{.N.Get2Y}}(p);
			let limited_abs = {{.N.Max}}({{.N.Min}}({{.N.Abs}}(x), {{.Two}}), {{.Half}});
			let squared = {{.N.Mul}}(y, y);
			let trig = {{.N.Add}}({{.N.Cos}}({{.Zero}}), {{.N.Sin}}({{.HalfPi}}));
			let powers = {{.N.Add}}({{.N.Sqrt}}(squared), {{.N.Pow}}({{.Two}}, {{.Three}}));
			let raw = {{.N.Add}}({{.N.Add}}({{.N.Add}}(limited_abs, powers), {{.N.Add}}(trig, {{.N.Exp}}({{.Zero}}))), {{.N.Atan2}}({{.One}}, {{.Zero}}));
			return {{.N.Clamp}}(raw, {{.Zero}}, {{.Hundred}});
		`,
	)
	vals := ExecuteShapeKernel(
		t,
		kernelToNative(Fixed64Numerics, k),
		Vec2{-3, -4},
		Vec2{0.25, 0.5},
	)
	vals.ExpectFloats(t, []float32{17 + math.Pi/2, 12 + math.Pi/2}, 1e-4)
}

func TestFixed64NumericsVectorOps(t *testing.T) {
	k := testFixed64Kernel(
		SDF3D,
		"vector_ops",
		`
			let offset = {{.N.Make3}}({{.OnePointFive}}, {{.NegTwo}}, {{.Quarter}});
			let shifted = {{.N.Add3}}(p, offset);
			let scaled = {{.N.Scale3}}(shifted, {{.Half}});
			let basis_x = {{.N.Make3}}({{.One}}, {{.Zero}}, {{.Zero}});
			let basis_y = {{.N.Make3}}({{.Zero}}, {{.One}}, {{.Zero}});
			let cross = {{.N.Cross3}}(basis_x, basis_y);
			return {{.N.Add}}({{.N.Dot3}}(scaled, scaled), {{.N.Get3Z}}(cross));
		`,
	)
	vals := ExecuteShapeKernel(
		t,
		kernelToNative(Fixed64Numerics, k),
		Vec3{0.5, 2, -0.25},
		Vec3{0, 0, 0},
	)
	vals.ExpectFloats(t, []float32{2, 2.578125}, 1e-4)
}

func TestFixed64NumericsVectorMathOps(t *testing.T) {
	k := testFixed64Kernel(
		SDF3D,
		"vector_math_ops",
		`
			let x = {{.N.Abs3}}(p);
			let y = {{.N.Sqrt3}}({{.N.Make3}}({{.One}}, {{.Four}}, {{.Nine}}));
			let mi = {{.N.Min3}}(x, y);
			let ma = {{.N.Max3}}(x, y);
			let powers = {{.N.Pow3}}({{.N.Make3}}({{.Two}}, {{.Three}}, {{.Four}}), {{.N.Make3}}({{.Two}}, {{.Two}}, {{.Half}}));
			let funcs = {{.N.Add3}}({{.N.Add3}}({{.N.Exp3}}({{.N.Make3}}({{.Zero}}, {{.Zero}}, {{.Zero}})), {{.N.Cos3}}({{.N.Make3}}({{.Zero}}, {{.Zero}}, {{.Zero}}))), {{.N.Sin3}}({{.N.Make3}}({{.HalfPi}}, {{.HalfPi}}, {{.HalfPi}})));
			return {{.N.Add}}({{.N.Add}}({{.N.Dot3}}(mi, ma), {{.N.Dot3}}(powers, {{.N.Make3}}({{.One}}, {{.One}}, {{.One}}))), {{.N.Dot3}}(funcs, {{.N.Make3}}({{.One}}, {{.One}}, {{.One}})));
		`,
	)
	vals := ExecuteShapeKernel(
		t,
		kernelToNative(Fixed64Numerics, k),
		Vec3{-2, 1, -0.25},
	)
	vals.ExpectFloats(t, []float32{28.75}, 1e-4)
}

func TestFixed64NumericsComparisons(t *testing.T) {
	k := testFixed64Kernel(
		Solid2D,
		"comparisons",
		`
			let lower = {{.N.Make2}}({{.NegOne}}, {{.NegOne}});
			let upper = {{.N.Make2}}({{.One}}, {{.One}});
			let in_range = all({{.N.Ge2}}(p, lower)) && all({{.N.Le2}}(p, upper));
			let not_on_corner = !all({{.N.Eq2}}(p, upper));
			let x = {{.N.Get2X}}(p);
			let y = {{.N.Get2Y}}(p);
			let ordered = {{.N.Lt}}(x, y) || {{.N.Eq}}(x, y) || {{.N.Gt}}(x, y);
			return in_range && not_on_corner && ordered;
		`,
	)
	vals := ExecuteShapeKernel(
		t,
		kernelToNative(Fixed64Numerics, k),
		Vec2{0, 0},
		Vec2{-1, 0.5},
		Vec2{1, 1},
		Vec2{1.25, 0},
	)
	vals.ExpectBools(t, []bool{true, true, false, false})
}

func TestFixed64NumericsLengths(t *testing.T) {
	k := testFixed64Kernel(
		SDF2D,
		"lengths",
		`
			let unit_x = {{.N.Make2}}({{.One}}, {{.Zero}});
			return {{.N.Add}}({{.N.Len2}}(p), {{.N.Dist2}}(p, unit_x));
		`,
	)
	vals := ExecuteShapeKernel(
		t,
		kernelToNative(Fixed64Numerics, k),
		Vec2{3, 4},
		Vec2{0, 0},
	)
	vals.ExpectFloats(t, []float32{float32(5 + math.Sqrt(20)), 1}, 1e-4)
}

func testFixed64Kernel(kind ShapeKind, name string, body string) ShapeKernel {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, name)
	return ShapeKernel{
		Kind: kind,
		IDs:  ids,
		Code: WGSL(
			`
				fn {{.Entrypoint}}(p: {{.ArgType}}) -> {{.ReturnType}} {
					{{.Body}}
				}
			`,
			"Entrypoint", entrypointName,
			"ArgType", kind.ArgType(Fixed64Numerics),
			"ReturnType", kind.ReturnType(Fixed64Numerics),
			"Body", WGSL(
				body,
				"N", Fixed64Numerics.Symbols,
				"Zero", Fixed64Numerics.Symbols.Zero,
				"One", Fixed64Numerics.Symbols.One,
				"NegOne", Fixed64Numerics.Literal(-1),
				"NegTwo", Fixed64Numerics.Literal(-2),
				"Quarter", Fixed64Numerics.Literal(0.25),
				"Half", Fixed64Numerics.Literal(0.5),
				"HalfPi", Fixed64Numerics.Literal(math.Pi/2),
				"OnePointFive", Fixed64Numerics.Literal(1.5),
				"Two", Fixed64Numerics.Literal(2),
				"Three", Fixed64Numerics.Literal(3),
				"Four", Fixed64Numerics.Literal(4),
				"Nine", Fixed64Numerics.Literal(9),
				"Hundred", Fixed64Numerics.Literal(100),
			),
		),
		EntrypointName: entrypointName,
	}
}
