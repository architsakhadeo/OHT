package util

import (
	"fmt"

	"github.com/stellentus/tile"
)

type MultiTiler struct {
	// it is a tiler which calculates indices.
	it tile.IndexTiler

	// scalers is an array of scaling information to be applied to each input data dimension.
	scalers []Scaler

	// numIndices is the number of tile indicies that could be produced. Valid indices range between 0 and numOutputs-1.
	numIndices int
}

// NewMultiTiler creates a new Indexing Tiler, which returns a slice of indexes based on the tiles' hashes.
// Input data is scaled based on the provided scaling structs.
// Then each dimension is tiled individually and in pairs.
// The length of the []Scaler should either be equal to 'numDims', or it should be length 1 (in which case it will apply to all tiles).
func NewMultiTiler(numDims, tiles int, scalers []Scaler) (MultiTiler, error) {
	if numDims < 1 {
		return MultiTiler{}, fmt.Errorf("A tile coder must have a positive number of dimensions, not %d", numDims)
	}

	if len(scalers) == 1 && numDims > 1 {
		// Fill the array with copies of the si
		si := scalers[0]
		scalers = make([]Scaler, numDims)
		for i := range scalers {
			scalers[i] = si
		}
	}
	if len(scalers) != numDims {
		return MultiTiler{}, fmt.Errorf("Mismatch between len(Scaler) and the number of dimensions (%d != %d)", len(scalers), numDims)
	}

	numIndices := 0

	// Create tilers for each single dimension
	singles, err := tile.NewSinglesTiler(numDims, tiles)
	if err != nil {
		return MultiTiler{}, fmt.Errorf("Could not create aggregate tiler: %s", err.Error())
	}
	for _, scaler := range scalers {
		numIndices += maxIndices(scaler.MaxRange, 1, tiles)
	}

	// Create tilers for each pair of dimensions
	pairs, err := tile.NewPairsTiler(numDims, tiles)
	if err != nil {
		return MultiTiler{}, fmt.Errorf("Could not create aggregate tiler: %s", err.Error())
	}
	for _, scaler := range scalers {
		numIndices += maxIndices(scaler.MaxRange, 2, tiles)
	}

	// Create a mega-tiler that appends pairs and singles
	til, err := tile.NewAggregateTiler([]tile.Tiler{singles, pairs})
	if err != nil {
		return MultiTiler{}, fmt.Errorf("Could not create aggregate tiler: %s", err.Error())
	}

	// Now change the tiler to return indexes instead of hashes
	it, err := tile.NewIndexingTiler(til, numIndices)
	return MultiTiler{it: it, scalers: scalers, numIndices: numIndices}, err
}

// Tile returns a vector of length equal to tiles (the argument to NewMultiTiler). That vector contains indices
// describing the input data.
func (st MultiTiler) Tile(data []float64) ([]int, error) {
	if len(data) != len(st.scalers) {
		return nil, fmt.Errorf("Cannot tile data of different length than []Scaler (%d != %d)", len(data), len(st.scalers))
	}

	for i, si := range st.scalers {
		data[i] = si.Scale(data[i])
	}

	indices := st.it.Tile(data)

	return indices, st.it.CheckError()
}

func (st MultiTiler) NumberOfIndices() int {
	return st.numIndices
}

// maxIndices returns the maximum number of indices with a given maximum range of input data, number of dimensions
// in the input data, and number of tilings. It assumes the number of tilings is the same in each dimension.
// For example, when tiling a 4-dimensional input, with each input value ranging from -3 to 6, and 32 tilings, the
// call would be `maxIndices(6-(-3), 4, 32)`.
func maxIndices(maxRange, numDims, numTilings int) int {
	ranges := make([]int, numDims)
	for i := range ranges {
		ranges[i] = maxRange
	}
	return maxIndicesForRanges(ranges, numTilings)
}

// maxIndicesForRanges returns the maximum number of indices with a given maximum range of input data and number of
// tilings. The ranges slice contains the maximum range for each of the input values being hashed.
func maxIndicesForRanges(ranges []int, numTilings int) int {
	result := 1
	for _, val := range ranges {
		// Add +1 because the most values along the "edges" of the region will include tiles that are otherwise entirely out of the region.
		result *= val + 1
	}

	result *= numTilings
	return result
}
