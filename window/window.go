// Package window provides various windowing functions for use in designing FIR
// filters.
package window

import "math"

// Func is a windowing function.
type Func func(x float64, n int) float64

// Blackman is a Blackman windowing function.
//
// Reference: https://en.wikipedia.org/wiki/Window_function#Blackman_window
func Blackman(x float64, n int) float64 {
	return 0.42 - (0.5 * math.Cos((2*math.Pi*x)/float64(n))) + (0.08 * math.Cos((4*math.Pi*x)/float64(n)))
}

// Hann is a Hann windowing function.
//
// Reference: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
func Hann(x float64, n int) float64 {
	return hannHamming(0.5, x, n)
}

// Hamming is a Hamming windowing function.
//
// Reference: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
func Hamming(x float64, n int) float64 {
	return hannHamming(0.53836, x, n)
}

func hannHamming(a, x float64, n int) float64 {
	return a - (1-a)*math.Cos(2*x*math.Pi/float64(n))
}

// Lanczos is a Lanczos windowing function.
//
// Reference: https://en.wikipedia.org/wiki/Window_function#Lanczos_window
func Lanczos(x float64, n int) float64 {
	return Sinc((2 * x / float64(n)) - 1)
}

// Bartlett is a Bartlett windowing function.
//
// Reference: https://en.wikipedia.org/wiki/Window_function#Triangular_window
func Bartlett(x float64, n int) float64 {
	return 1 - 2*math.Abs(x-float64(n)/2)/float64(n)
}

// Sinc is the cardinal sinc function. Use it to create other window functions.
//
// Reference: https://en.wikipedia.org/wiki/Sinc_function
func Sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}
