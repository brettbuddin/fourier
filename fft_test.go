package fourier

import (
	"fmt"
	"math"
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

func TestFrequencyDomainZeroPaddingResample(t *testing.T) {
	for _, tt := range []struct {
		src      []float64
		scale    int
		expected []float64
	}{
		{
			src:      []float64{1, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5},
			scale:    1,
			expected: []float64{1, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5},
		},
		{
			src:      []float64{1, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5},
			scale:    2,
			expected: []float64{1, 0.75, 0.5, 0.75, 1.0, 0.75, 0.5, 0.75, 1.0, 0.75, 0.5, 0.75, 1.0, 0.75, 0.5, 0.75},
		},
		{
			src:      []float64{1, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5},
			scale:    8,
			expected: []float64{1, 0.9809698831278217, 0.9267766952966369, 0.8456708580912724, 0.75, 0.6543291419087276, 0.5732233047033631, 0.5190301168721783, 0.5, 0.5190301168721783, 0.5732233047033631, 0.6543291419087276, 0.75, 0.8456708580912724, 0.9267766952966369, 0.9809698831278217, 1, 0.9809698831278217, 0.9267766952966369, 0.8456708580912724, 0.75, 0.6543291419087276, 0.5732233047033631, 0.5190301168721783, 0.5, 0.5190301168721783, 0.5732233047033631, 0.6543291419087276, 0.75, 0.8456708580912724, 0.9267766952966369, 0.9809698831278217, 1, 0.9809698831278217, 0.9267766952966369, 0.8456708580912724, 0.75, 0.6543291419087276, 0.5732233047033631, 0.5190301168721783, 0.5, 0.5190301168721783, 0.5732233047033631, 0.6543291419087276, 0.75, 0.8456708580912724, 0.9267766952966369, 0.9809698831278217, 1, 0.9809698831278217, 0.9267766952966369, 0.8456708580912724, 0.75, 0.6543291419087276, 0.5732233047033631, 0.5190301168721783, 0.5, 0.5190301168721783, 0.5732233047033631, 0.6543291419087276, 0.75, 0.8456708580912724, 0.9267766952966369, 0.9809698831278217},
		},
	} {
		var (
			l       = len(tt.src)
			ln      = l * tt.scale
			csrc    = make([]complex128, l)
			cpadded = make([]complex128, ln)
			out     = make([]float64, ln)
		)

		for i, v := range tt.src {
			csrc[i] = complex(v*float64(tt.scale), 0)
		}
		require.NoError(t, Forward(csrc))

		// Split the spectral content down the middle and copy both ends into the
		// ends of a new upscaled buffer. This will leave the zeros in the center.
		for i := 0; i < l/2; i++ {
			cpadded[i] = csrc[i]
			cpadded[len(cpadded)-1-i] = csrc[len(csrc)-1-i]
		}
		require.NoError(t, Inverse(cpadded))
		for i, v := range cpadded {
			out[i] = real(v)
		}

		assert.Equal(t, tt.expected, out)
	}
}

func TestMagnitude(t *testing.T) {
	const fc = 10.0
	const fs = 32.0 * fc
	const size = 256

	carrier := make([]complex128, size)
	for i := 0; i < len(carrier); i++ {
		v := math.Cos((float64(i) * 2 * math.Pi * fc) / fs)
		carrier[i] = complex(v, 0)
	}

	err := Forward(carrier)
	require.NoError(t, err)

	abs := make([]float64, len(carrier))
	err = Magnitude(abs, carrier)
	require.NoError(t, err)

	// Carrier Frequency = 10Hz
	// Resolution = 1.25Hz
	// Spike @ Carrier Frequency / Resolution = 8
	assert.Equal(t, 1.0, abs[8])

	// Spike in negative frequencies
	assert.Equal(t, 1.0, abs[248])

	// Other parts should be close to zero
	assert.Equal(t, 0.0, math.Round(abs[0]))
	assert.Equal(t, 0.0, math.Round(abs[10]))

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
