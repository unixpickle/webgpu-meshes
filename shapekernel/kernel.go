package shapekernel

import (
	"fmt"
	"strings"
)

type ShapeKind int

const (
	Solid2D ShapeKind = iota
	Solid3D
	SDF2D
	SDF3D
)

func (s ShapeKind) ReturnType() string {
	switch s {
	case Solid2D, Solid3D:
		return "bool"
	case SDF2D, SDF3D:
		return "f32"
	}
	panic("unknown ShapeKind")
}

func (s ShapeKind) ArgType() string {
	return fmt.Sprintf("vec%d<f32>", s.Dim())
}

func (s ShapeKind) Dim() int {
	switch s {
	case Solid2D, SDF2D:
		return 2
	case Solid3D, SDF3D:
		return 3
	}
	panic("unknown ShapeKind")
}

type IDTracker struct {
	NextFnID     int
	NextBufferID int
}

type Buffer struct {
	Name        string
	Constructor func() []float32
}

type ShapeKernel struct {
	Kind           ShapeKind
	IDs            IDTracker
	Buffers        []Buffer
	Code           string
	EntrypointName string
}

// ShiftIDs updates all of the IDs within the code of k
// to start at the new given offsets.
func ShiftIDs(k ShapeKernel, offsets IDTracker) ShapeKernel {
	k.Buffers = append([]Buffer{}, k.Buffers...)
	for i := k.IDs.NextBufferID - 1; i >= 0; i-- {
		old := fmt.Sprintf("sym_buf_%d_", i)
		repl := fmt.Sprintf("sym_buf_%d_", i+offsets.NextBufferID)
		k.Code = strings.ReplaceAll(k.Code, old, repl)
		for i, b := range k.Buffers {
			b.Name = strings.ReplaceAll(b.Name, old, repl)
			k.Buffers[i] = b
		}
	}
	for i := k.IDs.NextFnID - 1; i >= 0; i-- {
		old := fmt.Sprintf("sym_fn_%d_", i)
		repl := fmt.Sprintf("sym_fn_%d_", i+offsets.NextFnID)
		k.Code = strings.ReplaceAll(k.Code, old, repl)
		k.EntrypointName = strings.ReplaceAll(k.EntrypointName, old, repl)
	}
	k.IDs.NextBufferID += offsets.NextBufferID
	k.IDs.NextFnID += offsets.NextFnID
	return k
}

func genFunctionID(idt *IDTracker, name string) string {
	result := fmt.Sprintf("sym_fn_%d_%s", idt.NextFnID, name)
	idt.NextFnID += 1
	return result
}

func genBufferID(idt *IDTracker, name string) string {
	result := fmt.Sprintf("sym_buf_%d_%s", idt.NextBufferID, name)
	idt.NextBufferID += 1
	return result
}
