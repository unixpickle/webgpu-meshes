package shapekernel

import (
	"math"
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
	Metaball2D
	Metaball3D
	FalloffFunc
)

func (s ShapeKind) ReturnType(n Numerics) string {
	switch s {
	case Solid2D, Solid3D:
		return "bool"
	case SDF2D, SDF3D, Metaball2D, Metaball3D, FalloffFunc:
		return n.Symbols.Dtype
	}
	panic("unknown ShapeKind")
}

func (s ShapeKind) ArgType(n Numerics) string {
	if s == FalloffFunc {
		return n.Symbols.Dtype
	}
	if s.Dim() == 2 {
		return n.Symbols.Dtype2
	} else if s.Dim() == 3 {
		return n.Symbols.Dtype3
	} else {
		panic("no dtype for vec dimension")
	}
}

func (s ShapeKind) Dim() int {
	switch s {
	case Solid2D, SDF2D, Metaball2D:
		return 2
	case Solid3D, SDF3D, Metaball3D:
		return 3
	}
	panic("unknown ShapeKind")
}

type IDTracker struct {
	NextFnID     int
	NextBufferID int
}

type Buffer struct {
	WGSLType    string
	Name        string
	Constructor func() []uint32
}

func Float32Buffer(name string, constructor func() []float32) Buffer {
	return Buffer{
		WGSLType: "f32",
		Name:     name,
		Constructor: func() []uint32 {
			return float32SliceToBits(constructor())
		},
	}
}

func Uint32Buffer(name string, constructor func() []uint32) Buffer {
	return Buffer{
		WGSLType:    "u32",
		Name:        name,
		Constructor: constructor,
	}
}

func Int32Buffer(name string, constructor func() []int32) Buffer {
	return Buffer{
		WGSLType: "i32",
		Name:     name,
		Constructor: func() []uint32 {
			return int32SliceToUint32(constructor())
		},
	}
}

func float32SliceToBits(values []float32) []uint32 {
	result := make([]uint32, len(values))
	for i, x := range values {
		result[i] = math.Float32bits(x)
	}
	return result
}

func int32SliceToUint32(values []int32) []uint32 {
	result := make([]uint32, len(values))
	for i, x := range values {
		result[i] = uint32(x)
	}
	return result
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
	result := "sym_fn_" + strconv.Itoa(idt.NextFnID) + "_" + name
	idt.NextFnID += 1
	return result
}

func genBufferID(idt *IDTracker, name string) string {
	result := "sym_buf_" + strconv.Itoa(idt.NextBufferID) + "_" + name
	idt.NextBufferID += 1
	return result
}
