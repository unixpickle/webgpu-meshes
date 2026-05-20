package shapekernel

import "github.com/unixpickle/model3d/model2d"

const mesh2DBVHLeafSize = 4

type mesh2DBVH struct {
	Segments   []float32
	NodeBounds []float32
	NodeData   []uint32
	Height     int
}

type mesh2DBVHNode struct {
	Min        [2]float32
	Max        [2]float32
	Start      uint32
	Count      uint32
	RightChild uint32
	SkipIndex  uint32
}

type mesh2DBVHBuilder struct {
	Segments []*model2d.Segment
	Nodes    []mesh2DBVHNode
}

func newMesh2DBVH(m *model2d.Mesh) mesh2DBVH {
	segs := append([]*model2d.Segment{}, m.SegmentSlice()...)
	if len(segs) > 1 {
		model2d.GroupSegments(segs)
	}

	builder := &mesh2DBVHBuilder{Segments: segs}
	height := 0
	if len(segs) > 0 {
		_, _, _, height = builder.BuildRange(0, len(segs))
	}

	return mesh2DBVH{
		Segments:   flattenMesh2DSegments(segs),
		NodeBounds: builder.FlattenBounds(),
		NodeData:   builder.FlattenData(),
		Height:     height,
	}
}

func flattenMesh2DSegments(segs []*model2d.Segment) []float32 {
	result := make([]float32, 0, len(segs)*4)
	for _, seg := range segs {
		result = append(result,
			float32(seg[0].X), float32(seg[0].Y),
			float32(seg[1].X), float32(seg[1].Y),
		)
	}
	return result
}

func (m *mesh2DBVHBuilder) BuildRange(start, end int) (int, model2d.Coord, model2d.Coord, int) {
	nodeIdx := len(m.Nodes)
	m.Nodes = append(m.Nodes, mesh2DBVHNode{})

	if end-start <= mesh2DBVHLeafSize {
		min, max := m.BoundsForRange(start, end)
		m.Nodes[nodeIdx] = mesh2DBVHNode{
			Min:       coord2DToFloat32(min),
			Max:       coord2DToFloat32(max),
			Start:     uint32(start),
			Count:     uint32(end - start),
			SkipIndex: uint32(nodeIdx + 1),
		}
		return nodeIdx, min, max, 1
	}

	mid := start + (end-start)/2
	_, leftMin, leftMax, leftHeight := m.BuildRange(start, mid)
	rightIdx, rightMin, rightMax, rightHeight := m.BuildRange(mid, end)
	min := leftMin.Min(rightMin)
	max := leftMax.Max(rightMax)
	height := leftHeight
	if rightHeight > height {
		height = rightHeight
	}

	m.Nodes[nodeIdx] = mesh2DBVHNode{
		Min:        coord2DToFloat32(min),
		Max:        coord2DToFloat32(max),
		RightChild: uint32(rightIdx),
		SkipIndex:  uint32(len(m.Nodes)),
	}
	return nodeIdx, min, max, height + 1
}

func (m *mesh2DBVHBuilder) BoundsForRange(start, end int) (model2d.Coord, model2d.Coord) {
	min := m.Segments[start].Min()
	max := m.Segments[start].Max()
	for _, seg := range m.Segments[start+1 : end] {
		min = min.Min(seg.Min())
		max = max.Max(seg.Max())
	}
	return min, max
}

func (m *mesh2DBVHBuilder) FlattenBounds() []float32 {
	result := make([]float32, 0, len(m.Nodes)*4)
	for _, node := range m.Nodes {
		result = append(result,
			node.Min[0], node.Min[1],
			node.Max[0], node.Max[1],
		)
	}
	return result
}

func (m *mesh2DBVHBuilder) FlattenData() []uint32 {
	result := make([]uint32, 0, len(m.Nodes)*4)
	for _, node := range m.Nodes {
		result = append(result, node.Start, node.Count, node.RightChild, node.SkipIndex)
	}
	return result
}

func coord2DToFloat32(c model2d.Coord) [2]float32 {
	return [2]float32{float32(c.X), float32(c.Y)}
}
