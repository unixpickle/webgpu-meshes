package shapekernel

import (
	"fmt"
)

func Translate(k ShapeKernel, offset Vector) ShapeKernel {
	fnName := genFunctionID(&k.IDs, "translate")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> %s {
				let newP = p - %s;
				return %s(newP);
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.Kind.ReturnType(),
		offset.WebGPUVec(),
		k.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}

func Scale(k ShapeKernel, scales Vector) ShapeKernel {
	fnName := genFunctionID(&k.IDs, "scale")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> %s {
				let newP = p / %s
				return %s(newP);
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.Kind.ReturnType(),
		scales.WebGPUVec(),
		k.EntrypointName,
	)
	k.EntrypointName = fnName
	return k
}
