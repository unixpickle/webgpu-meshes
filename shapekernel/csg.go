package shapekernel

import (
	"fmt"
	"strings"
)

// UnionSolids takes the union of one or more solids.
func UnionSolids(solids []ShapeKernel) ShapeKernel {
	return solidBooleanOp(solids, "||", "union")
}

// IntersectSolids takes the intersection of one or more solids.
func IntersectSolids(solids []ShapeKernel) ShapeKernel {
	return solidBooleanOp(solids, "&&", "intersection")
}

func solidBooleanOp(solids []ShapeKernel, op, name string) ShapeKernel {
	if len(solids) == 0 {
		panic("expected at least one solid")
	} else if len(solids) == 1 {
		return solids[0]
	}

	for i := 1; i < len(solids); i++ {
		if solids[i].Kind != solids[0].Kind {
			panic("mismatching shape kinds")
		}
	}

	k := solids[0]
	k.Buffers = append([]Buffer{}, k.Buffers...)
	orCode := []string{fmt.Sprintf("%s(p)", k.EntrypointName)}
	for i := 1; i < len(solids); i++ {
		nextK := ShiftIDs(solids[i], k.IDs)
		k.IDs = nextK.IDs
		k.Buffers = append(k.Buffers, nextK.Buffers...)
		k.Code += "\n" + nextK.Code
		orCode = append(orCode, fmt.Sprintf("%s(p)", nextK.EntrypointName))
	}

	fnName := genFunctionID(&k.IDs, name+"_solid")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> bool {
				return %s;
			}
		`),
		fnName,
		k.Kind.ArgType(),
		strings.Join(orCode, " "+op+" "),
	)
	k.EntrypointName = fnName
	return k
}
