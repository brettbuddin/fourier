package filter

import (
	"testing"

	"github.com/brettbuddin/fourier/window"
	"github.com/stretchr/testify/require"
)

func TestLowPass(t *testing.T) {
	kernel := make([]float64, 10)
	MakeLowPass(kernel, window.Blackman, 0.5)

	expected := []float64{
		-5.409800153010306e-34,
		-1.5675664736656203e-18,
		7.826365172768749e-18,
		-1.9872378605190914e-17,
		3.310443906868464e-17,
		1,
		3.310443906868464e-17,
		-1.9872378605190917e-17,
		7.826365172768752e-18,
		-1.567566473665619e-18,
	}
	require.InEpsilonSlice(t, expected, kernel, 1e-10)
}

func TestHighPass(t *testing.T) {
	kernel := make([]float64, 10)
	MakeHighPass(kernel, window.Blackman, 0.5)

	expected := []float64{
		5.409800153010306e-34,
		1.5675664736656203e-18,
		-7.826365172768749e-18,
		1.9872378605190914e-17,
		-3.310443906868464e-17,
		-1,
		-3.310443906868464e-17,
		1.9872378605190917e-17,
		-7.826365172768752e-18,
		1.567566473665619e-18,
	}
	require.InEpsilonSlice(t, expected, kernel, 1e-10)
}

func TestBandPass(t *testing.T) {
	kernel := make([]float64, 10)
	MakeBandPass(kernel, window.Blackman, 0.25, 0.5)

	expected := []float64{
		8.852297468639933e-19,
		-7.822375291978086e-19,
		0.02134438446523165,
		-2.982816317717476e-17,
		-0.27085135668587773,
		0.49901394444129243,
		-0.27085135668587773,
		-2.982816317717477e-17,
		0.021344384465231663,
		-7.822375291978078e-19,
	}
	require.InEpsilonSlice(t, expected, kernel, 1e-10)
}

func TestBandReject(t *testing.T) {
	kernel := make([]float64, 10)
	MakeBandReject(kernel, window.Blackman, 0.25, 0.5)

	expected := []float64{
		-8.852297468639933e-19,
		7.822375291978086e-19,
		-0.02134438446523165,
		2.982816317717476e-17,
		0.27085135668587773,
		-0.49901394444129243,
		0.27085135668587773,
		2.982816317717477e-17,
		-0.021344384465231663,
		7.822375291978078e-19,
	}
	require.InEpsilonSlice(t, expected, kernel, 1e-10)
}
