package fourier

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var epsilon = 0.0000001

func TestBitReversal(t *testing.T) {
	b16 := "%016b"
	require.Equal(t, "0000000000000100", fmt.Sprintf(b16, reverseBits(2, 4)))
	require.Equal(t, "0000000001000000", fmt.Sprintf(b16, reverseBits(2, 8)))
	require.Equal(t, "0000000010000000", fmt.Sprintf(b16, reverseBits(1, 8)))
	require.Equal(t, "0000000000100000", fmt.Sprintf(b16, reverseBits(1, 6)))
	require.Equal(t, "1000000000000000", fmt.Sprintf(b16, reverseBits(1, 16)))
	require.Equal(t, "0000000000000001", fmt.Sprintf(b16, reverseBits(1, 1)))
	require.Equal(t, "0000000000000001", fmt.Sprintf(b16, reverseBits(1, 0)))
}

func TestLog2(t *testing.T) {
	require.Equal(t, uint(1), log2(3))
	require.Equal(t, uint(4), log2(16))
	require.Equal(t, uint(5), log2(32))
}

func TestIsPowerOf2(t *testing.T) {
	require.True(t, isPowerOfTwo(2))
	require.True(t, isPowerOfTwo(4))
	require.True(t, isPowerOfTwo(8))
	require.True(t, isPowerOfTwo(1))
	require.False(t, isPowerOfTwo(5))
	require.False(t, isPowerOfTwo(7))
}

func TestNextPowerOfTwo(t *testing.T) {
	require.Equal(t, 1, nextPowerOfTwo(1))
	require.Equal(t, 8, nextPowerOfTwo(5))
	require.Equal(t, 64, nextPowerOfTwo(50))
	require.Equal(t, 64, nextPowerOfTwo(64))
}

func TestButterflyReorder(t *testing.T) {
	buf := []complex128{
		complex(1, 0),
		complex(2, 0),
		complex(3, 0),
		complex(4, 0),
		complex(5, 0),
		complex(6, 0),
		complex(7, 0),
		complex(8, 0),
	}

	reorder(buf)

	require.Equal(t, []complex128{
		complex(1, 0),
		complex(5, 0),
		complex(3, 0),
		complex(7, 0),
		complex(2, 0),
		complex(6, 0),
		complex(4, 0),
		complex(8, 0),
	}, buf)
}

func TestForwardTransform(t *testing.T) {
	buf := make([]complex128, 8)
	for i := 0; i < 4; i++ {
		buf[i] = complex(1, 0)
	}

	Forward(buf)
	cmplxEqualEpsilon(t, []complex128{
		(4 + 0i),
		(1 - 2.414213562373095i),
		(0 + 0i),
		(1 - 0.4142135623730949i),
		(0 + 0i),
		(0.9999999999999999 + 0.4142135623730949i),
		(0 + 0i),
		(0.9999999999999997 + 2.414213562373095i),
	}, buf, epsilon)
}

func TestRoundTripTransform(t *testing.T) {
	buf := make([]complex128, 4)
	for i := range buf {
		buf[i] = complex(float64(i)+1, 0)
	}

	Forward(buf)
	cmplxEqualEpsilon(t, []complex128{
		(10 + 0i),
		(-2 + 2i),
		(-2 + 0i),
		(-2 - 2i),
	}, buf, epsilon)

	Inverse(buf)
	cmplxEqualEpsilon(t, []complex128{
		(1 + 0i),
		(2 + 0i),
		(3 + 0i),
		(4 + 0i),
	}, buf, epsilon)
}

func cmplxEqualEpsilon(t *testing.T, expected, actual []complex128, epsilon float64) {
	t.Helper()

	for i := range expected {
		if real(expected[i]) == 0 {
			assert.Equal(t, 0.0, real(actual[i]))
		} else {
			assert.InEpsilon(t, real(expected[i]), real(actual[i]), epsilon)
		}

		if imag(expected[i]) == 0 {
			assert.Equal(t, 0.0, imag(actual[i]))
		} else {
			assert.InEpsilon(t, imag(expected[i]), imag(actual[i]), epsilon)
		}
	}
}

func BenchmarkFFT(b *testing.B) {
	b.ReportAllocs()
	b.StopTimer()
	src := make([]complex128, 64)
	for i := range src {
		src[i] = complex(float64(i)+1, 0)
	}

	buf := make([]complex128, len(src))
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		b.StartTimer()
		Forward(buf)
		b.StopTimer()
	}
}

func BenchmarkIFFT(b *testing.B) {
	b.ReportAllocs()
	b.StopTimer()
	src := make([]complex128, 64)
	for i := range src {
		src[i] = complex(float64(i)+1, 0)
	}
	Forward(src)

	buf := make([]complex128, len(src))
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		b.StartTimer()
		Inverse(buf)
		b.StopTimer()
	}
}
