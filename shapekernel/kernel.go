package shapekernel

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

type ShapeKind int

var offsetRegexps = map[string]*regexp.Regexp{
	"sym_fn_":  regexp.MustCompile("sym_fn_([0-9]*)_"),
	"sym_buf_": regexp.MustCompile("sym_buf_([0-9]*)_"),
}

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
	if offsets.NextBufferID > 0 {
		k.Code = offsetSymbolNumbers(k.Code, "sym_buf_", offsets.NextBufferID)
		for i, b := range k.Buffers {
			b.Name = offsetSymbolNumbers(b.Name, "sym_buf_", offsets.NextBufferID)
			k.Buffers[i] = b
		}
	}
	k.Code = offsetSymbolNumbers(k.Code, "sym_fn_", offsets.NextFnID)
	k.EntrypointName = offsetSymbolNumbers(k.EntrypointName, "sym_fn_", offsets.NextFnID)
	k.IDs.NextBufferID += offsets.NextBufferID
	k.IDs.NextFnID += offsets.NextFnID
	return k
}

func offsetSymbolNumbers(code string, prefix string, offset int) string {
	expr := offsetRegexps[prefix]
	matches := expr.FindAllStringSubmatchIndex(code, -1)
	parts := make([]string, 0, len(matches)*2+1)
	lastEnd := 0
	for _, submatch := range matches {
		parts = append(parts, code[lastEnd:submatch[0]])
		lastEnd = submatch[1]

		idx, err := strconv.Atoi(code[submatch[2]:submatch[3]])
		if err != nil {
			panic(err)
		}
		parts = append(parts, prefix+strconv.Itoa(idx+offset)+"_")
	}
	parts = append(parts, code[lastEnd:])
	return strings.Join(parts, "")
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
