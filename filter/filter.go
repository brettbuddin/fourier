// Package filter provides builders for designing kernels for common filter
// types.
package filter

import (
	"math"

	"github.com/brettbuddin/fourier/window"
)

// MakeLowPass creates a low-pass filter impulse response. It filters
// frequencies higher than the cutoff frequency.
func MakeLowPass(h []float64, wf window.Func, cutoff float64) {
	n := len(h)
	for i := range h {
		x := 2 * math.Pi * cutoff
		if i == n/2 {
			h[i] = x
		} else {
			y := float64(i) - float64(n)/2
			h[i] = (math.Sin(x*y) / y) * wf(float64(i), n)
		}
	}
	normalize(h)
}

// MakeHighPass creates a high-pass filter impulse response. It filters
// frequencies lower than the cutoff frequency.
func MakeHighPass(h []float64, wf window.Func, cutoff float64) {
	MakeLowPass(h, wf, cutoff)
	for i := range h {
		h[i] = -h[i]
	}
}

// MakeBandReject creates a band-reject filter impulse response. It filters out
// frequencies between the two stop frequencies.
func MakeBandReject(h []float64, wf window.Func, stop1, stop2 float64) {
	a := make([]float64, len(h))
	b := make([]float64, len(h))
	MakeLowPass(a, wf, stop1)
	MakeHighPass(b, wf, stop2)
	for i := range h {
		h[i] = a[i] + b[i]
	}
}

// MakeBandPass creates a band-pass filter impulse response. It allows
// frequences between the two stop frequencies.
func MakeBandPass(h []float64, wf window.Func, stop1, stop2 float64) {
	MakeBandReject(h, wf, stop1, stop2)
	for i := range h {
		h[i] = -h[i]
	}
}

func normalize(w []float64) {
	var sum float64
	for i := range w {
		sum += w[i]
	}
	scale := 1.0 / sum
	for i := range w {
		w[i] *= scale
	}
}
