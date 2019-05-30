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
	buf := make([]complex128, 4)
	for i := range buf {
		buf[i] = complex(float64(i)+1, 0)
	}

	Forward(buf)

	expected := []complex128{
		complex(10, 0),
		complex(-2, 2),
		complex(-2, 0),
		complex(-2, -2),
	}

	for i := range expected {
		if real(expected[i]) == 0 {
			assert.Equal(t, 0.0, real(buf[i]))
		} else {
			assert.InEpsilon(t, real(expected[i]), real(buf[i]), epsilon)
		}

		if imag(expected[i]) == 0 {
			assert.Equal(t, 0.0, imag(buf[i]))
		} else {
			assert.InEpsilon(t, imag(expected[i]), imag(buf[i]), epsilon)
		}
	}
}

func TestRoundTripTransform(t *testing.T) {
	buf := make([]complex128, 4)
	for i := range buf {
		buf[i] = complex(float64(i)+1, 0)
	}

	Forward(buf)
	Inverse(buf)

	expected := []float64{1, 2, 3, 4}

	for i := range expected {
		assert.InEpsilon(t, expected[i], real(buf[i]), epsilon)
	}
}
