package shapekernel

type Vec2 [2]float32

func (v Vec2) WebGPUVec() string {
	return WGSL("vec2<f32>({{.X}}, {{.Y}})", "X", v[0], "Y", v[1])
}

func (v Vec2) Dim() int {
	return 2
}

func (v Vec2) At(i int) float32 {
	return v[i]
}

type Vec3 [3]float32

func (v Vec3) WebGPUVec() string {
	return WGSL("vec3<f32>({{.X}}, {{.Y}}, {{.Z}})", "X", v[0], "Y", v[1], "Z", v[2])
}

func (v Vec3) Dim() int {
	return 3
}

func (v Vec3) At(i int) float32 {
	return v[i]
}

type Segment3 [2]Vec3

type Vector interface {
	Dim() int
	WebGPUVec() string
	At(i int) float32
}
