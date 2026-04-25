package shapekernel

import "testing"

func TestOffsetSymbolNumbers(t *testing.T) {
	output := offsetSymbolNumbers(Dedent(`
		sym_fn_0_baz(x) {
			return sym_fn_1_bar(y);
		}
	`), "sym_fn_", 5)
	expected := Dedent(`
		sym_fn_5_baz(x) {
			return sym_fn_6_bar(y);
		}
	`)
	if output != expected {
		t.Fatalf("expected %s but got %s", expected, output)
	}
}
