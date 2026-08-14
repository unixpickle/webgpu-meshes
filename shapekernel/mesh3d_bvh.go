package shapekernel

import "github.com/unixpickle/model3d/model3d"

const mesh3DBVHLeafSize = 4

type mesh3DBVH struct {
	Triangles  []float32
	NodeBounds []float32
	NodeData   []uint32
	Height     int
}

type mesh3DBVHNode struct {
	Min        [3]float32
	Max        [3]float32
	Start      uint32
	Count      uint32
	RightChild uint32
	SkipIndex  uint32
}

type mesh3DBVHBuilder struct {
	Triangles []*model3d.Triangle
	Nodes     []mesh3DBVHNode
}

// newMesh3DBVH creates a BVH from already-grouped triangles.
func newMesh3DBVH(tris []*model3d.Triangle) mesh3DBVH {
	builder := &mesh3DBVHBuilder{Triangles: tris}
	height := 0
	if len(tris) > 0 {
		_, _, _, height = builder.BuildRange(0, len(tris))
	}

	return mesh3DBVH{
		Triangles:  flattenMesh3DTriangles(tris),
		NodeBounds: builder.FlattenBounds(),
		NodeData:   builder.FlattenData(),
		Height:     height,
	}
}

func flattenMesh3DTriangles(tris []*model3d.Triangle) []float32 {
	result := make([]float32, 0, len(tris)*9)
	for _, tri := range tris {
		for _, c := range tri {
			result = append(result, float32(c.X), float32(c.Y), float32(c.Z))
		}
	}
	return result
}

func (m *mesh3DBVHBuilder) BuildRange(start, end int) (int, model3d.Coord3D, model3d.Coord3D, int) {
	nodeIdx := len(m.Nodes)
	m.Nodes = append(m.Nodes, mesh3DBVHNode{})

	if end-start <= mesh3DBVHLeafSize {
		min, max := m.BoundsForRange(start, end)
		m.Nodes[nodeIdx] = mesh3DBVHNode{
			Min:       coord3DToFloat32(min),
			Max:       coord3DToFloat32(max),
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

	m.Nodes[nodeIdx] = mesh3DBVHNode{
		Min:        coord3DToFloat32(min),
		Max:        coord3DToFloat32(max),
		RightChild: uint32(rightIdx),
		SkipIndex:  uint32(len(m.Nodes)),
	}
	return nodeIdx, min, max, height + 1
}

func (m *mesh3DBVHBuilder) BoundsForRange(start, end int) (model3d.Coord3D, model3d.Coord3D) {
	min := m.Triangles[start].Min()
	max := m.Triangles[start].Max()
	for _, tri := range m.Triangles[start+1 : end] {
		min = min.Min(tri.Min())
		max = max.Max(tri.Max())
	}
	return min, max
}

func (m *mesh3DBVHBuilder) FlattenBounds() []float32 {
	result := make([]float32, 0, len(m.Nodes)*6)
	for _, node := range m.Nodes {
		result = append(result,
			node.Min[0], node.Min[1], node.Min[2],
			node.Max[0], node.Max[1], node.Max[2],
		)
	}
	return result
}

func (m *mesh3DBVHBuilder) FlattenData() []uint32 {
	result := make([]uint32, 0, len(m.Nodes)*4)
	for _, node := range m.Nodes {
		result = append(result, node.Start, node.Count, node.RightChild, node.SkipIndex)
	}
	return result
}

func coord3DToFloat32(c model3d.Coord3D) [3]float32 {
	return [3]float32{float32(c.X), float32(c.Y), float32(c.Z)}
}
