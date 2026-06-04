package shapekernel

type Vec2 [2]float64

func (v Vec2) WebGPUVec(n Numerics) string {
	return WGSL(
		"{{.N.Make2}}({{.X}}, {{.Y}})",
		"N", n.Symbols,
		"X", n.Literal(v[0]), "Y", n.Literal(v[1]),
	)
}

func (v Vec2) Dim() int {
	return 2
}

func (v Vec2) At(i int) float64 {
	return v[i]
}

type Vec3 [3]float64

func (v Vec3) WebGPUVec(n Numerics) string {
	return WGSL(
		"{{.N.Make3}}({{.X}}, {{.Y}}, {{.Z}})",
		"N", n.Symbols,
		"X", n.Literal(v[0]), "Y", n.Literal(v[1]), "Z", n.Literal(v[2]),
	)
}

func (v Vec3) Dim() int {
	return 3
}

func (v Vec3) At(i int) float64 {
	return v[i]
}

type Segment3 [2]Vec3

type Vector interface {
	Dim() int
	WebGPUVec(Numerics) string
	At(i int) float64
}
