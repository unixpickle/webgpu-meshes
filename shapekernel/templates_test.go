package shapekernel

import (
	"strings"
	"testing"
)

func TestTemplateFormatsFloatLiterals(t *testing.T) {
	got := Template("{{.Int32}} {{.Int64}} {{.Frac}} {{.Exp}} {{.NegZero}}",
		"Int32", float32(1),
		"Int64", float64(2),
		"Frac", float32(1.25),
		"Exp", float32(1e-6),
		"NegZero", float32(-0.0),
	)

	for _, want := range []string{"1.0", "2.0", "1.25", "0.0"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected formatted WGSL float literal %q in %q", want, got)
		}
	}
	if !strings.Contains(got, "1e-06") {
		t.Fatalf("expected scientific-notation float literal in %q", got)
	}
}
