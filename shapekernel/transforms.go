package shapekernel

import (
	"fmt"
	"math"
)

func Translate(k ShapeKernel, offset Vector) ShapeKernel {
	if k.Kind == FalloffFunc {
		panic("cannot translate falloff functions")
	}
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
	if k.Kind == FalloffFunc {
		panic("cannot scale falloff functions")
	}
	scaleCode := ""
	if k.Kind == SDF2D || k.Kind == SDF3D {
		scaleCode = fmt.Sprintf(" / %f", math.Abs(float64(scales.At(0))))
	}
	fnName := genFunctionID(&k.IDs, "scale")
	k.Code += "\n" + fmt.Sprintf(
		Dedent(`
			fn %s(p: %s) -> %s {
				let newP = p / %s;
				return %s(newP)%s;
			}
		`),
		fnName,
		k.Kind.ArgType(),
		k.Kind.ReturnType(),
		scales.WebGPUVec(),
		k.EntrypointName,
		scaleCode,
	)
	k.EntrypointName = fnName
	return k
}
