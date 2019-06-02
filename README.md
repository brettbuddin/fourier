# Fourier

[![GoDoc](https://godoc.org/github.com/brettbuddin/fourier?status.svg)](https://godoc.org/github.com/brettbuddin/fourier)
[![Go Report Card](https://goreportcard.com/badge/github.com/brettbuddin/fourier)](https://goreportcard.com/report/github.com/brettbuddin/fourier)
[![Coverage Status](https://codecov.io/gh/brettbuddin/fourier/graph/badge.svg)](https://codecov.io/gh/brettbuddin/fourier)
[![Build Status](https://travis-ci.org/brettbuddin/fourier.svg)](https://travis-ci.org/brettbuddin/fourier)

- Fast Fourier Transform implementation via [Cooley-Tukey (Radix-2 DIT)](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm).
- Convolution engine which performs partitioned convolution in the frequency domain using the [overlap-add method](https://en.wikipedia.org/wiki/Overlap–add_method).
- Windowing functions for creating impulse responses. (e.g.  Hann, Lanczos, etc)
- Functions for creating common types of FIR filters. (e.g.  low-pass, high-pass, etc)

## Examples

### Fast Fourier Transform

```go
buf := make([]complex128, 8)
for i := range buf {
    buf[i] = complex(float64(i + 1), 0)
}
// buf [(1+0i) (2+0i) (3+0i) (4+0i) (5+0i) (6+0i) (7+0i) (8+0i)]
fourier.Forward(buf)
// buf [(36+0i) (-4+9.65685424949238i) (-4+4i) (-4+1.6568542494923797i)...
fourier.Inverse(buf)
// buf [(1+0i) (2+0i) (3+0i) (4+0i) (5+0i) (6+0i) (7+0i) (8+0i)] (+/- small error)
```

### Convolution

```go
var (
    ir       = []float64{1, 1}
    conv, _  = fourier.NewConvolver(8, ir)
    in       = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    out      = make([]float64, len(in)+len(ir)-1)
)

_ = conv.Convolve(out, in, len(out))

// out [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 16] (+/- small error)
```

### Filtering

```go
var (
    blockSize  = 256
    sampleRate = 44100.0
    cutoff     = 200.0
    kernel     = make([]float64, 32)
)

// Build a filter kernel that filters frequencies higher than 200Hz at 44.1kHz sampling rate.
filter.MakeLowPass(kernel, window.Lanczos, cutoff/sampleRate)

conv, _ := fourier.NewConvolver(blockSize, kernel)

conv.Convolve(out, in, len(out))
```
