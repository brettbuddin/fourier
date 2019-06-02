package fourier

import (
	"errors"
	"fmt"
	"math"
)

// Maximum impulse response length of 10 seconds at 96kHz
const maxIRSamples = 20 * 96000

// Convolver performs partioned convolution using the overlap-add method. It is
// designed to convolve very long input streams with a FIR filter.
type Convolver struct {
	// Sizes
	blockSize, fftSize, numImpulseSegments, numInputSegments int

	// Buffers
	inputSegments, impulseSegments [][]complex128
	outputBuf, tempBuf             []complex128
	inputBuf, overlapBuf           []float64

	// Internal state
	inputSegmentPos, inputPos int
	channel, numChannels      int
}

// NewConvolver returns a new Convolver.
//
// blockSize is maximum size consumed from the input (loaded into an input
// segment) for each convolution operation (FFT -> multiply -> IFFT). Ideally,
// it's maximum number of samples you plan on processing for each call to Convolve.
//
// The length of the impulse response is limited to 1920000 samples (20s * 96kHz).
// Exceeding this maximum length will result in truncation of the IR.
func NewConvolver(blockSize int, ir []float64, opts ...ConvolverOption) (*Convolver, error) {
	if blockSize == 0 {
		return nil, errors.New("block size cannot be zero")
	}

	// Determine the sizes of the buffers
	blockSize = nextPowerOfTwo(blockSize)
	fftSize := 4 * blockSize
	if blockSize > 128 {
		fftSize = 2 * blockSize
	}

	// Allocate buffers
	var (
		inputBuf   = make([]float64, fftSize)
		outputBuf  = make([]complex128, fftSize)
		tempBuf    = make([]complex128, fftSize)
		overlapBuf = make([]float64, fftSize)
		c          = &Convolver{
			blockSize:   blockSize,
			fftSize:     fftSize,
			numChannels: 1,

			inputBuf:   inputBuf,
			outputBuf:  outputBuf,
			tempBuf:    tempBuf,
			overlapBuf: overlapBuf,
		}
	)

	for _, opt := range opts {
		if err := opt(c); err != nil {
			return c, err
		}
	}

	err := c.SetImpulseResponse(ir)

	return c, err
}

// SetImpulseResponse sets the impulse response used in convolution.
func (c *Convolver) SetImpulseResponse(ir []float64) error {
	if len(ir) == 0 {
		return errors.New("impulse response length cannot be zero")
	}

	var (
		fftSize            = c.fftSize
		blockSize          = c.blockSize
		irSize             = min(len(ir), maxIRSamples)
		numImpulseSegments = int(irSize/(fftSize-blockSize) + 1)
	)

	numInputSegments := numImpulseSegments
	if blockSize <= 128 {
		numInputSegments = 3 * numImpulseSegments
	}

	// Allocate input segment buffers
	inputSegments := make([][]complex128, numInputSegments)
	for i := range inputSegments {
		inputSegments[i] = make([]complex128, fftSize)
	}

	// Allocate impulse segment buffers
	impulseSegments := make([][]complex128, numImpulseSegments)
	for i := range impulseSegments {
		impulseSegments[i] = make([]complex128, fftSize)
	}

	// Split the impulse response into segments and transform each segment to
	// the frequency domain.
	for i := 0; i < numImpulseSegments; i++ {
		if i == 0 {
			impulseSegments[i][0] = complex(1, 0)
		}

		for j := 0; j < fftSize-blockSize; j++ {
			irIdx := j + i*(fftSize-blockSize)
			if irIdx < len(ir) {
				v := ir[irIdx]
				if math.IsNaN(v) {
					v = 0
				}
				impulseSegments[i][j] = complex(v, 0)
			}
		}

		Forward(impulseSegments[i])
	}

	c.numInputSegments = numInputSegments
	c.numImpulseSegments = numImpulseSegments
	c.inputSegments = inputSegments
	c.impulseSegments = impulseSegments

	return nil
}

// Convolve convolves an a chunk of input against the loaded impulse response.
func (c *Convolver) Convolve(out, in []float64, numSamples int) error {
	var (
		numImpulseSegments  = c.numImpulseSegments
		numInputSegments    = c.numInputSegments
		channel             = c.channel
		numChannels         = c.numChannels
		step                = numInputSegments / numImpulseSegments
		fftSize             = c.fftSize
		blockSize           = c.blockSize
		numSamplesProcessed = 0
	)

	for numSamplesProcessed < numSamples {
		var (
			numRemaining        = numSamples - numSamplesProcessed
			blockLimit          = blockSize - c.inputPos
			numSamplesToProcess = min(numRemaining, blockLimit)
		)

		// Copy the input into the internal input buffer. Fill with zeros if
		// we've stepped beyond the length of the input.
		for i := 0; i < numSamplesToProcess; i++ {
			inIdx := channel + numSamplesProcessed + i*numChannels
			var v float64
			if inIdx <= len(in)-1 {
				v = in[inIdx]
			}
			c.inputBuf[c.inputPos+i] = v
		}
		inputSegment := c.inputSegments[c.inputSegmentPos]
		if err := cmplxCopyReal(inputSegment, c.inputBuf); err != nil {
			return err
		}

		// Forward FFT
		if err := Forward(inputSegment); err != nil {
			return err
		}

		// Multiply
		if c.inputPos == 0 {
			cmplxZero(c.tempBuf)

			index := c.inputSegmentPos
			for i := 1; i < numImpulseSegments; i++ {
				index += step
				if index >= numInputSegments {
					index -= numInputSegments
				}

				inputSegment := c.inputSegments[index]
				impulseSegment := c.impulseSegments[i]

				cmplxMultiplyAdd(c.tempBuf, inputSegment, impulseSegment)
			}
		}

		if err := cmplxCopy(c.outputBuf, c.tempBuf); err != nil {
			return err
		}
		if err := cmplxMultiplyAdd(c.outputBuf, inputSegment, c.impulseSegments[0]); err != nil {
			return err
		}

		// Inverse FFT
		if err := Inverse(c.outputBuf); err != nil {
			return err
		}

		// Add overlap to the output
		for i := 0; i < numSamplesToProcess; i++ {
			var (
				outIdx = numSamplesProcessed + channel + i*numChannels
				pos    = c.inputPos + i
			)
			// Guard against stepping outside the bounds of the output buffer if
			// the user supplied a buffer that's not a multiple of the block
			// size.
			if outIdx > len(out)-1 {
				continue
			}
			out[outIdx] = real(c.outputBuf[pos]) + c.overlapBuf[pos]
		}

		c.inputPos += numSamplesToProcess

		if c.inputPos == blockSize {
			c.inputPos = 0
			zero(c.inputBuf)

			// Additional overlap when segment size > block size
			arErr := cmplxAddReal(c.outputBuf[blockSize:], c.overlapBuf[blockSize:], fftSize-2*blockSize)
			if arErr != nil {
				return arErr
			}

			// Save overlap
			for i := 0; i < fftSize-blockSize; i++ {
				c.overlapBuf[i] = real(c.outputBuf[i+blockSize])
			}

			// Step the current segment backwards
			if c.inputSegmentPos > 0 {
				c.inputSegmentPos--
			} else {
				c.inputSegmentPos = numInputSegments - 1
			}
		}

		numSamplesProcessed += numSamplesToProcess
	}

	return nil
}

// cmplxMultiplyAdd multiplies two complex buffers and adds the result to another.
func cmplxMultiplyAdd(dest, a, b []complex128) error {
	var (
		la    = len(a)
		lb    = len(b)
		ldest = len(dest)
	)
	if ldest != la || ldest != lb {
		return fmt.Errorf("buffer sizes do not match: dest=%d a=%d b=%d", ldest, la, lb)
	}
	for i := range dest {
		dest[i] += a[i] * b[i]
	}
	return nil
}

// cmplxCopy copies one complex buffer into another.
func cmplxCopy(dest, src []complex128) error {
	var (
		ldest = len(dest)
		lsrc  = len(src)
	)
	if ldest != lsrc {
		return fmt.Errorf("buffer lengths do not match: dest=%d src=%d", ldest, lsrc)
	}
	for i := range src {
		dest[i] = src[i]
	}
	return nil
}

// cmplxCopyReal copies a real buffer to a complex buffer. The imaginary
// component is set to zero.
func cmplxCopyReal(dest []complex128, src []float64) error {
	var (
		ldest = len(dest)
		lsrc  = len(src)
	)
	if ldest != lsrc {
		return fmt.Errorf("buffer lengths do not match: dest=%d src=%d", ldest, lsrc)
	}
	for i := range src {
		dest[i] = complex(src[i], 0)
	}
	return nil
}

// cmplxAddReal adds a real buffer to a complex buffer's real component.
func cmplxAddReal(dest []complex128, vals []float64, size int) error {
	var (
		ldest = len(dest)
		lvals = len(vals)
	)
	if size > ldest || size > lvals {
		return fmt.Errorf("operation size larger than buffers: dest=%d vals=%d size=%d", ldest, lvals, size)
	}
	for i := range vals {
		dest[i] += complex(vals[i], imag(dest[i]))
	}
	return nil
}

// cmplxZero zeros out a complex buffer.
func cmplxZero(dest []complex128) {
	for i := range dest {
		dest[i] = complex(0, 0)
	}
}

// zero zeros out a buffer.
func zero(dest []float64) {
	for i := range dest {
		dest[i] = 0
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ConvolverOption is a configuration option for Convolver.
type ConvolverOption func(*Convolver) error

// ForChannel configures a Convolver to target a specific channel when the input
// buffer contains multiple interleaved channels.
func ForChannel(channel, numChannels int) ConvolverOption {
	return func(c *Convolver) error {
		if channel < 0 {
			return errors.New("channel cannot be negative")
		}
		if numChannels < 1 {
			return errors.New("number of channels cannot be less than 1")
		}
		if channel >= numChannels {
			return errors.New("channel out of range of total number of channels")
		}
		c.channel = channel
		c.numChannels = numChannels
		return nil
	}
}
