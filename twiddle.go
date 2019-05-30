package fourier

import "math"

// table is a trigonometric "twiddle" table.
type table struct {
	sin, cos []float64
}

var twiddleTables = map[int]*table{}

// twiddleTable looks up a twiddle table for a particular FFT size. If the table
// has already been calculated, a cached version is returned.
func twiddleTable(size int) *table {
	if _, ok := twiddleTables[size]; ok {
		return twiddleTables[size]
	}
	t := &table{
		cos: make([]float64, size/2),
		sin: make([]float64, size/2),
	}
	for i := 0; i < size/2; i++ {
		fi := float64(i)
		fsize := float64(size)
		t.cos[i] = math.Cos(2 * math.Pi * fi / fsize)
		t.sin[i] = math.Sin(2 * math.Pi * fi / fsize)
	}
	twiddleTables[size] = t
	return t
}
