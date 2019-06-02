// Package fourier provides a Fast Fourier Transform implementation and
// Convolver that performs partioned convolution in the frequency domain.
package fourier

import (
	"errors"
	"math/cmplx"
)

// Forward performs a forward FFT via Cooley-Tukey Radix-2 DIT. The buffer
// length is required to be a power of two.
func Forward(v []complex128) error {
	return forward(v)
}

// Inverse performs an inverse FFT via Cooley-Tukey Radix-2 DIT. The buffer
// length is required to be a power of two.
func Inverse(v []complex128) error {
	for i := range v {
		v[i] = cmplx.Conj(v[i])
	}
	if err := forward(v); err != nil {
		return err
	}
	normalize(v)
	return nil
}

// normalize proportions the values to the length of the buffer
func normalize(v []complex128) {
	scale := 1 / float64(len(v))
	for i := range v {
		v[i] = complex(real(v[i])*scale, 0)
	}
}

// forward performs a forward FFT.
func forward(v []complex128) error {
	n := len(v)
	if n == 2 {
		return nil
	}

	if !isPowerOfTwo(n) {
		return errors.New("buffer length is not a power of two")
	}

	table := twiddleTable(n)

	// Reorder the input in preparation for the Butterfly
	reorder(v)

	// Perform butterfly
	for size := 2; size <= n; size *= 2 {
		var (
			half = size / 2
			step = n / size
		)

		for i := 0; i < n; i += size {
			var (
				j = i
				k = 0
			)
			for j < i+half {
				var (
					l       = j + half
					cos     = table.cos[k]
					sin     = table.sin[k]
					twiddle = complex(real(v[l])*cos+imag(v[l])*sin, -real(v[l])*sin+imag(v[l])*cos)
				)

				v[l] = v[j] - twiddle
				v[j] += twiddle

				j++
				k += step
			}
		}
	}
	return nil
}

// reorder reorders a complex buffer's values to form the pattern necessary for
// the Cooley-Tukey radix-2 DIT butterfly operation.
func reorder(v []complex128) {
	var (
		lv   = uint(len(v))
		bits = log2(lv)
		i    uint
	)
	for ; i < lv; i++ {
		j := reverseBits(i, bits)
		if j > i {
			v[j], v[i] = v[i], v[j]
		}
	}
}

// log2 returns log base-2 of an integer
func log2(v uint) uint {
	var r uint
	for v >>= 1; v != 0; v >>= 1 {
		r++
	}
	return r
}

// reverseBits reverses all bits up until a designated place significance.
func reverseBits(v, bits uint) uint {
	if bits == 1 {
		return v
	}

	var r uint = v & 1
	bits--

	for v >>= 1; v != 0; v >>= 1 {
		r <<= 1
		r |= v & 1
		bits--
	}
	return r << bits
}

// isPowerOfTwo determines whether or not an integer is a power of two.
func isPowerOfTwo(v int) bool {
	return (v != 0) && (v&(v-1)) == 0
}

// nextPowerOfTwo finds the next power of two above the value provided.
func nextPowerOfTwo(v int) int {
	if isPowerOfTwo(v) {
		return v
	}

	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v++
	return v
}
