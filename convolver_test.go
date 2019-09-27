package fourier

import (
	"fmt"
	"math"
	"testing"

	"github.com/brettbuddin/fourier/filter"
	"github.com/brettbuddin/fourier/window"
	"github.com/stretchr/testify/require"
)

func TestConvolution_SmallerImpulse(t *testing.T) {
	var (
		blockSize   = 8
		impulseSize = 4

		impulse = make([]float64, impulseSize)
		input   = make([]float64, blockSize)
		output  = make([]float64, blockSize)
	)

	for i := range impulse {
		impulse[i] = 1
	}

	for i := range input {
		input[i] = float64(i) + 1
	}

	convolver, err := NewConvolver(blockSize, impulse)
	require.NoError(t, err)

	cErr := convolver.Convolve(output, input, blockSize)
	require.NoError(t, cErr)

	expected := []float64{1, 3, 6, 10, 14, 18, 22, 26}

	require.InEpsilonSlice(t, expected, output, epsilon)
}

func TestConvolution_LargerImpulse(t *testing.T) {
	var (
		blockSize   = 8
		inputSize   = 8
		impulseSize = 256

		impulse = make([]float64, impulseSize)
		input   = make([]float64, inputSize)
		output  = make([]float64, inputSize+impulseSize-1)
	)

	for i := range impulse {
		impulse[i] = 1
	}
	impulse[len(impulse)-1] = 2

	for i := range input {
		input[i] = 1
	}

	convolver, err := NewConvolver(blockSize, impulse)
	require.NoError(t, err)

	// Convolve in blocks, comparing the output of each block to an expected
	// value. The last value of the convolution will be different in delta from
	// all the rest.
	for i := 0; i < len(output); i += blockSize {
		var (
			inBegin = min(i, len(input))
			inEnd   = min(i+blockSize, len(input))
			outEnd  = min(i+blockSize, len(output))
		)
		cErr := convolver.Convolve(output[i:outEnd], input[inBegin:inEnd], blockSize)
		require.NoError(t, cErr)
	}

	expected := []float64{1, 2, 3, 4, 5, 6, 7}
	for j := 0; j < 248; j++ {
		expected = append(expected, 8)
	}
	expected = append(expected, []float64{9, 8, 7, 6, 5, 4, 3, 2}...)
	require.InEpsilonSlice(t, expected, output, epsilon)
}

func TestConvolution_PartialOfMaxBlockSize(t *testing.T) {
	var (
		partialSize = 4
		blockSize   = 8
		impulseSize = 16

		impulse = make([]float64, impulseSize)
		input   = make([]float64, partialSize)
		output  = make([]float64, partialSize)
	)

	for i := range impulse {
		impulse[i] = 1
	}

	for i := range input {
		input[i] = 1
	}

	convolver, err := NewConvolver(blockSize, impulse)
	require.NoError(t, err)

	cErr := convolver.Convolve(output, input, partialSize)
	require.NoError(t, cErr)

	expected := []float64{1, 2, 3, 4}
	require.InEpsilonSlice(t, expected, output, epsilon)
}

func TestConvolver_ErroneousCreation(t *testing.T) {
	_, lErr := NewConvolver(0, []float64{1})
	require.Error(t, lErr)

	_, irErr := NewConvolver(64, []float64{})
	require.Error(t, irErr)
}

func BenchmarkConvolver(b *testing.B) {
	var (
		blockSize = 64
		ir        = make([]float64, 500)
		in        = make([]float64, blockSize)
		out       = make([]float64, blockSize)
	)

	conv, _ := NewConvolver(blockSize, ir)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		conv.Convolve(out, in, blockSize)
	}
}

func TestConvolution_Interleaved(t *testing.T) {
	var (
		blockSize   = 8
		impulseSize = 4
		numChannels = 2

		impulse = make([]float64, impulseSize)
		input   = make([]float64, numChannels*blockSize)
		output  = make([]float64, numChannels*blockSize)
	)

	for i := range impulse {
		impulse[i] = 1
	}

	for i := 0; i < len(input); i += numChannels {
		input[i] = float64(i/numChannels) + 1
		input[i+1] = 2 * input[i]
	}

	// Create a convolver for each channel and convolve that channel's input
	for i := 0; i < numChannels; i++ {
		conv, err := NewConvolver(blockSize, impulse, ForChannel(i, numChannels))
		require.NoError(t, err)

		cErr := conv.Convolve(output, input, blockSize)
		require.NoError(t, cErr)
	}

	expected := []float64{1, 2, 3, 6, 6, 12, 10, 20, 14, 28, 18, 36, 22, 44, 26, 52}

	require.InEpsilonSlice(t, expected, output, epsilon)
}

func ExampleConvolver_simple() {
	var (
		blockSize = 8
		ir        = []float64{1, 1}
		conv, _   = NewConvolver(blockSize, ir)
		in        = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		out       = make([]float64, len(in)+len(ir)-1)
	)

	// Convolve the entire input buffer. We're using the output buffer length
	// here in order to capture the full convolved output of INPUT_LENGTH +
	// IR_LENGTH - 1. Convolving more samples will result in the values
	// eventually dropping to zero.
	_ = conv.Convolve(out, in, len(out))

	// Round to nearest integer (removing error) for pretty printing
	roundTo(out, 1e10)

	fmt.Println(out)
	// Output: [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 16]
}

func ExampleConvolver_chunks() {
	var (
		blockSize = 8
		ir        = []float64{1, 1}
		conv, _   = NewConvolver(blockSize, ir)
		in        = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		out       = make([]float64, len(in)+len(ir)-1)
	)

	for i := 0; i < len(out); i += blockSize {
		var (
			inBegin = min(i, len(in))
			inEnd   = min(i+blockSize, len(in))
			outEnd  = min(i+blockSize, len(out))

			inChunk  = in[inBegin:inEnd]
			outChunk = out[i:outEnd]
		)

		// We are deriving the input and output chunks here, but they would be
		// presented to you via the callback mechanisms in a streaming audio
		// scenario. The algorithm is able to accomodate this chunking.
		_ = conv.Convolve(outChunk, inChunk, blockSize)
	}

	// Round to nearest integer (removing error) for pretty printing
	roundTo(out, 1e10)

	fmt.Println(out)
	// Output: [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 16]
}

func ExampleConvolver_filtering() {
	var (
		blockSize  = 256
		sampleRate = 320.0
		cutoff     = 30.0
		kernel     = make([]float64, 32)
		in         = make([]float64, blockSize)
	)

	// Sum two cosine waves, one at 10Hz and another at 90Hz
	carrier(in, 10.0, sampleRate)
	carrier(in, 90.0, sampleRate)

	// Calculate bins with high magnitude before filtering
	spikesBefore := detectSpikes(in)

	// Build a filter kernel that filters frequencies higher than 30Hz at 320Hz
	// sampling rate and convolve the summed signal with it.
	filter.MakeLowPass(kernel, window.Lanczos, cutoff/sampleRate)
	conv, _ := NewConvolver(blockSize, kernel)

	out := make([]float64, blockSize)
	conv.Convolve(out, in, len(out))

	// Calculate bins with high magnitude after filtering
	spikesAfter := detectSpikes(out)

	fmt.Println("spikes at (before):", spikesBefore)
	fmt.Println("spikes at (after):", spikesAfter)
	// Output: spikes at (before): [8 72 184 248]
	// spikes at (after): [8 248]
}

func magnitude(src []float64) []float64 {
	dest := make([]float64, len(src))
	freq := make([]complex128, len(src))
	for i, v := range src {
		freq[i] = complex(v, 0)
	}
	Forward(freq)
	Magnitude(dest, freq)
	return dest
}

func detectSpikes(buf []float64) []int {
	var spikes []int
	for i, v := range magnitude(buf) {
		if v > 0.2 {
			spikes = append(spikes, i)
		}
	}
	return spikes
}

func carrier(dest []float64, fc, fs float64) {
	for i := 0; i < len(dest); i++ {
		dest[i] += math.Cos((float64(i) * 2 * math.Pi * fc) / fs)
	}
}

func cmplxCarrier(dest []complex128, fc, fs float64) {
	destf := make([]float64, len(dest))
	carrier(destf, fc, fs)
	for i := range dest {
		dest[i] = complex(destf[i], 0)
	}
}

func roundTo(out []float64, epsilon float64) {
	for i := range out {
		out[i] = math.Round(out[i]*epsilon) / epsilon
	}
}
