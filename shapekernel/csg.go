package shapekernel

import (
	"fmt"
)

func UnionSolid(k1, k2 ShapeKernel) ShapeKernel {
	if k1.Kind != k2.Kind {
		panic("mismatching kinds passed to UnionSolid()")
	}
	k2 = ShiftIDs(k2, k1.IDs)

	k := k1
	k.Buffers = append(append([]Buffer{}, k1.Buffers...), k2.Buffers...)
	k.Code += "\n" + k2.Code
	k.IDs = k2.IDs

	fnName := genFunctionID(&k.IDs, "union_solid")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> bool {
				return %s(p) || %s(p);
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k1.EntrypointName,
		k2.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}
