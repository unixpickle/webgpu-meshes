package shapekernel

import (
	"bytes"
	"fmt"
	"text/template"
)

func TV(args ...any) map[string]any {
	if len(args)%2 != 0 {
		panic("template values require key/value pairs")
	}
	result := map[string]any{}
	for i := 0; i < len(args); i += 2 {
		key, ok := args[i].(string)
		if !ok {
			panic(fmt.Sprintf("template key at index %d is not a string", i))
		}
		result[key] = args[i+1]
	}
	return result
}

func Template(src string, args ...any) string {
	tmpl := template.Must(template.New("wgsl").Option("missingkey=error").Parse(src))
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, TV(args...)); err != nil {
		panic(err)
	}
	return buf.String()
}

func WGSL(src string, args ...any) string {
	return Template(Dedent(src), args...)
}

func AppendWGSL(k *ShapeKernel, src string, args ...any) {
	k.Code += "\n" + WGSL(src, args...)
}
