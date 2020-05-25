import tiles3 as tc



def ExampleIndexingTiler_Tile():
	print("\nExampleIndexingTiler_Tile:")
	# This reproduces the test by the same name in go, though indices 1 and 2 are swapped.
	# Output:
	# The indices for 4.99 are [0, 1, 2]
	# The indices for 5.32 are [3, 1, 2]
	# The indices for 5.34 are [3, 1, 4]
	# The indices for 5.5 are [3, 1, 4]

	numTilings = 3
	maxRange = 1
	ihtSize = int(numTilings * (maxRange + 1))
	iht = tc.IHT(ihtSize)
	testData = [4.99, 5.32, 5.34, 5.5]

	for val in testData:
		tiles = tc.tiles(iht, numTilings, [val])
		print("The indices for", val, "are", tiles)


class AggregateTiler:
	def __init__(self, maxRange, numTilingsArray):
		super().__init__()

		self.numTilings = numTilingsArray

		self.tilers = []
		self.ihtSize = []
		for i, numTilings in enumerate(numTilingsArray):
			self.ihtSize.append(int(numTilings * (maxRange + 1)))
			self.tilers.append(tc.IHT(self.ihtSize[i]))

		return

	def tile(self, vals):
		resultArrays = []

		for i, tiler in enumerate(self.tilers):
			resultArrays.append(tc.tiles(tiler, self.numTilings[i], vals))

		finalOutput = []
		for i, resArray in enumerate(resultArrays):
			if i != 0:
				for j, val in enumerate(resArray):
					resArray[j] = val+self.ihtSize[i-1]
			finalOutput += resArray

		return finalOutput


def ExampleAggregateTiler_Tile():
	print("\nExampleAggregateTiler_Tile:")
	# This reproduces the test by the same name in go, showing the bug when the tiling isn't a power of 2.
	# For example, look at the test for [3.35, 4] and [3.35, 4.68]. Both produce the same output, even
	# though they should be in different tilings.
	# Output:
	# The index for [3, 4] is [0, 1, 2, 9, 10]
	# The index for [3.35, 4] is [0, 1, 3, 9, 10]
	# The index for [3.68, 4] is [0, 4, 3, 9, 11]
	# The index for [3, 4.35] is [0, 1, 2, 9, 10]
	# The index for [3.35, 4.35] is [0, 1, 3, 9, 10]
	# The index for [3.68, 4.35] is [0, 4, 3, 9, 11]
	# The index for [3, 4.68] is [0, 1, 2, 9, 12]
	# The index for [3.35, 4.68] is [0, 1, 3, 9, 12]
	# The index for [3.68, 4.68] is [0, 4, 3, 9, 13]
	til = AggregateTiler(1, [3, 2])
	test = [
		[3, 4],
		[3.35, 4],
		[3.68, 4],

		[3, 4.35],
		[3.35, 4.35],
		[3.68, 4.35],

		[3, 4.68],
		[3.35, 4.68],
		[3.68, 4.68],
	]
	for data in test:
		print("The index for", data, "is", til.tile(data))


def printArrayWithWidth2(arr):
	print('-'.join(f'{val:02}' for val in arr), end='  ')


def printRowVals(rowVals):
	elmPerRow = 4
	elements = len(rowVals[0])
	for rowNum in range((elements+elmPerRow-1)//elmPerRow):
		for thisRowVals in rowVals:
			startIdx = rowNum*elmPerRow
			printArrayWithWidth2(thisRowVals[startIdx:startIdx+elmPerRow])
		print("")


def ExampleGrid(numTilings):
	maxRange = 1
	ihtSize = int(numTilings * (maxRange + 1)**2)
	iht = tc.IHT(ihtSize)
	print(f"\nExampleGrid (numTilings:{numTilings}, maxIndices:{ihtSize}):")
	# This prints out the unit grid for numTilings.

	for i in range(numTilings+1):
		rowVals = []
		for j in range(numTilings+1):
			tiles = tc.tiles(iht, numTilings, [i/numTilings, j/numTilings])
			if numTilings > 4:
				rowVals.append(tiles)
			else:
				printArrayWithWidth2(tiles)

		if numTilings > 4:
			printRowVals(rowVals)
		print("")


if __name__ == '__main__':
	print("Tiles Test")

	ExampleIndexingTiler_Tile()
	ExampleAggregateTiler_Tile()

	ExampleGrid(4)
	ExampleGrid(8)
	ExampleGrid(16)

	# ht, _ := newUnlimitedIndexTiler(3)
	# for _, data := range [][]float64{{}} {
	# 	fmt.Println("The indices for", data, "are", ht.Tile(data))
	# }
	# // Output:
	# // The indices for [4.99] are [0 1 2]
	# // The indices for [5.32] are [3 1 2]
	# // The indices for [5.34] are [3 4 2]
	# // The indices for [5.5] are [3 4 2]
