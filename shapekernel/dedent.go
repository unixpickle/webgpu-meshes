package shapekernel

import (
	"strings"
	"unicode"
)

// Dedent removes a uniform leading indentation based on the first non-blank line,
// and trims leading/trailing blank lines.
func Dedent(s string) string {
	lines := strings.Split(s, "\n")

	// Helper: check if line is all whitespace
	isBlank := func(line string) bool {
		for _, r := range line {
			if !unicode.IsSpace(r) {
				return false
			}
		}
		return true
	}

	// Trim leading blank lines
	start := 0
	for start < len(lines) && isBlank(lines[start]) {
		start++
	}

	// Trim trailing blank lines
	end := len(lines)
	for end > start && isBlank(lines[end-1]) {
		end--
	}

	lines = lines[start:end]
	if len(lines) == 0 {
		return ""
	}

	// Find leading whitespace prefix from first non-blank line
	first := lines[0]
	prefixLen := 0
	for _, r := range first {
		if unicode.IsSpace(r) {
			prefixLen += len(string(r))
		} else {
			break
		}
	}
	prefix := first[:prefixLen]

	// Remove that exact prefix from all lines (if present)
	for i, line := range lines {
		if strings.HasPrefix(line, prefix) {
			lines[i] = line[prefixLen:]
		}
	}

	return strings.Join(lines, "\n")
}
