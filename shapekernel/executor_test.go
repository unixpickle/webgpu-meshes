package shapekernel

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

type KernelExecutionResult struct {
	Bools  []bool    `json:"bools,omitempty"`
	Floats []float32 `json:"floats,omitempty"`
}

func (k *KernelExecutionResult) UnmarshalJSON(data []byte) error {
	type kernelExecutionResultJSON struct {
		Bools     []bool    `json:"bools,omitempty"`
		Floats    []float32 `json:"floats,omitempty"`
		FloatBits []uint32  `json:"floatBits,omitempty"`
	}
	var decoded kernelExecutionResultJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	k.Bools = decoded.Bools
	if len(decoded.FloatBits) > 0 {
		k.Floats = make([]float32, len(decoded.FloatBits))
		for i, bits := range decoded.FloatBits {
			k.Floats[i] = math.Float32frombits(bits)
		}
	} else {
		k.Floats = decoded.Floats
	}
	return nil
}

func (k *KernelExecutionResult) ExpectBools(t *testing.T, v []bool) {
	if len(v) != len(k.Bools) {
		t.Fatalf("unexpected bools: %v (expected %v)", k.Bools, v)
	}
	for i, x := range v {
		a := k.Bools[i]
		if a != x {
			t.Fatalf("unexpected bools: %v (expected %v)", k.Bools, v)
		}
	}
}

func (k *KernelExecutionResult) ExpectFloats(t *testing.T, v []float32, eps float32) {
	if len(v) != len(k.Floats) {
		t.Fatalf("unexpected floats: %v (expected %v)", k.Floats, v)
	}
	for i, x := range v {
		a := k.Floats[i]
		if math.IsNaN(float64(x)) || math.IsNaN(float64(a)) {
			if !(math.IsNaN(float64(x)) && math.IsNaN(float64(a))) {
				t.Fatalf("unexpected floats: %v (expected %v)", k.Floats, v)
			}
			continue
		}
		if math.IsInf(float64(x), 0) || math.IsInf(float64(a), 0) {
			if !(math.IsInf(float64(x), 1) && math.IsInf(float64(a), 1)) &&
				!(math.IsInf(float64(x), -1) && math.IsInf(float64(a), -1)) {
				t.Fatalf("unexpected floats: %v (expected %v)", k.Floats, v)
			}
			continue
		}
		if math.Abs(float64(a-x)) > float64(eps) {
			t.Fatalf("unexpected floats: %v (expected %v)", k.Floats, v)
		}
	}
}

type kernelExecutorRequest struct {
	Dim           int                    `json:"dim"`
	ReturnType    string                 `json:"returnType"`
	Code          string                 `json:"code"`
	Entrypoint    string                 `json:"entrypoint"`
	Buffers       []kernelExecutorBuffer `json:"buffers"`
	Inputs        [][4]float32           `json:"inputs"`
	InputCount    int                    `json:"inputCount"`
	InputBinding  int                    `json:"inputBinding"`
	OutputBinding int                    `json:"outputBinding"`
}

type kernelExecutorBuffer struct {
	Name   string    `json:"name"`
	Values []float32 `json:"values"`
}

// ExecuteShapeKernel runs a ShapeKernel in Chromium WebGPU and returns either
// bools or floats depending on the kernel kind.
func ExecuteShapeKernel(t testing.TB, k ShapeKernel, inputs ...Vector) KernelExecutionResult {
	t.Helper()

	request := kernelExecutorRequest{
		Dim:           k.Kind.Dim(),
		ReturnType:    k.Kind.ReturnType(),
		Code:          k.Code,
		Entrypoint:    k.EntrypointName,
		Buffers:       make([]kernelExecutorBuffer, len(k.Buffers)),
		Inputs:        make([][4]float32, len(inputs)),
		InputCount:    len(inputs),
		InputBinding:  len(k.Buffers),
		OutputBinding: len(k.Buffers) + 1,
	}
	for i, buf := range k.Buffers {
		values := buf.Constructor()
		request.Buffers[i] = kernelExecutorBuffer{
			Name:   buf.Name,
			Values: values,
		}
	}
	for i, input := range inputs {
		request.Inputs[i] = marshalKernelInput(t, k.Kind.Dim(), input)
	}
	if len(inputs) == 0 {
		return KernelExecutionResult{
			Bools:  []bool{},
			Floats: []float32{},
		}
	}

	stdinData, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, "node", executorScriptPath(t))
	cmd.Dir = executorRepoRoot(t)
	cmd.Stdin = bytes.NewReader(stdinData)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		t.Fatalf("run WebGPU executor: %v\n%s", err, formatExecutorFailure(stderr.String(), stdout.String()))
	}

	var result KernelExecutionResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode executor output: %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}

	switch k.Kind {
	case Solid2D, Solid3D:
		if len(result.Bools) != len(inputs) {
			t.Fatalf("executor returned %d bool results for %d inputs", len(result.Bools), len(inputs))
		}
	case SDF2D, SDF3D, Metaball2D, Metaball3D:
		if len(result.Floats) != len(inputs) {
			t.Fatalf("executor returned %d float results for %d inputs", len(result.Floats), len(inputs))
		}
	default:
		t.Fatalf("unsupported shape kind %v", k.Kind)
	}

	return result
}

func marshalKernelInput(t testing.TB, wantDim int, input Vector) [4]float32 {
	t.Helper()

	switch v := input.(type) {
	case Vec2:
		if wantDim != 2 {
			t.Fatalf("kernel expects %dD inputs but got Vec2", wantDim)
		}
		return [4]float32{v[0], v[1], 0, 0}
	case Vec3:
		if wantDim != 3 {
			t.Fatalf("kernel expects %dD inputs but got Vec3", wantDim)
		}
		return [4]float32{v[0], v[1], v[2], 0}
	default:
		t.Fatalf("unsupported input vector type %T; use Vec2 or Vec3", input)
		return [4]float32{}
	}
}

func executorScriptPath(t testing.TB) string {
	t.Helper()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve executor script path: runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(filename), "testdata", "webgpu_executor.js")
}

func executorRepoRoot(t testing.TB) string {
	t.Helper()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve repo root: runtime.Caller failed")
	}
	return filepath.Dir(filepath.Dir(filename))
}

func formatExecutorFailure(stderr, stdout string) string {
	var details []string
	if strings.Contains(stderr, "Cannot find package 'playwright'") || strings.Contains(stdout, "Cannot find package 'playwright'") {
		details = append(details, "install JavaScript dependencies with `npm install` before running WebGPU-backed Go tests")
	}
	if strings.Contains(stderr, "Executable doesn't exist") || strings.Contains(stderr, "playwright install") {
		details = append(details, "install the Chromium test browser with `npx playwright install chromium`")
	}
	if len(details) == 0 {
		details = append(details, "stdout:\n"+stdout)
		details = append(details, "stderr:\n"+stderr)
		return strings.Join(details, "\n")
	}
	if strings.TrimSpace(stderr) != "" {
		details = append(details, "stderr:\n"+stderr)
	}
	if strings.TrimSpace(stdout) != "" {
		details = append(details, "stdout:\n"+stdout)
	}
	return fmt.Sprintf("%s", strings.Join(details, "\n"))
}

func TestExecuteShapeKernelNonFiniteFloats(t *testing.T) {
	ids := IDTracker{}
	entrypointName := genFunctionID(&ids, "nonfinite")
	k := ShapeKernel{
		Kind: SDF2D,
		IDs:  ids,
		Code: fmt.Sprintf(
			Dedent(`
				fn %s(p: vec2<f32>) -> f32 {
					return p.x / p.y;
				}
			`),
			entrypointName,
		),
		EntrypointName: entrypointName,
	}
	vals := ExecuteShapeKernel(
		t,
		k,
		Vec2{1, 0},
		Vec2{-1, 0},
		Vec2{0, 0},
	)
	vals.ExpectFloats(t, []float32{
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.NaN()),
	}, 0)
}
