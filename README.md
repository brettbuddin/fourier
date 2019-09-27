# Fourier

[![GoDoc](https://godoc.org/github.com/brettbuddin/fourier?status.svg)](https://godoc.org/github.com/brettbuddin/fourier)
[![Go Report Card](https://goreportcard.com/badge/github.com/brettbuddin/fourier)](https://goreportcard.com/report/github.com/brettbuddin/fourier)
[![Coverage Status](https://codecov.io/gh/brettbuddin/fourier/graph/badge.svg)](https://codecov.io/gh/brettbuddin/fourier)

- Fast Fourier Transform implementation via [Cooley-Tukey (Radix-2 DIT)](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm).
- Convolution engine which performs partitioned convolution in the frequency domain using the [overlap-add method](https://en.wikipedia.org/wiki/Overlap–add_method).
- Windowing functions for creating impulse responses. (e.g.  Hann, Lanczos, etc)
- Functions for creating common types of FIR filters. (e.g.  low-pass, high-pass, etc)

This library was written for use in a real-time audio context. `Convolver`
allocates all of its buffers up-front and `Forward`/`Inverse` (FFT/IFFT) operate
in-place. This is to avoid allocations in the hot-path. I've used this library
to implement convolution reverb and perform various types of filtering.

(Usage Examples)[https://godoc.org/github.com/brettbuddin/fourier#pkg-examples]
