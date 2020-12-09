package main

import (
	"flag"
	"fmt"
	"math"
	"runtime"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
	"gonum.org/v1/gonum/mat"
)

var (
	cpus = flag.Int("cpus", 2, "Number of CPUs")
)

func main() {
	flag.Parse()

	runtime.GOMAXPROCS(*cpus) // Limit the number of CPUs to the provided value (unchanged if the input is <1)

	debug := logger.NewDebug(logger.DebugConfig{}) // TODO create a debug

	numTimesteps := 1000000
	numRuns := 1
	hyperparams := [5]string{"tilings", "tiles", "lambda", "epsilon", "alpha"}
	lower := [len(hyperparams)]float64{0.5, 0.5, 0.0, 0.0, 0.0}
	upper := [len(hyperparams)]float64{7.5, 4.5, 1.0, 1.0, 10}

	//typeOfHyperparams := [5]string{"discrete", "discrete", "continuous", "continuous", "continuous"}
	discreteHyperparamsIndices := [2]int64{0, 1}
	discreteRanges := [][]float64{[]float64{1, 2, 4, 8, 16, 32}, []float64{1, 2, 4}}
	discreteMidRanges := [][]float64{[]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, []float64{1.5, 2.5, 3.5}}

	numSamples := 300
	percentElite := 0.5
	numElite := int64(float64(numSamples) * percentElite)
	e := math.Pow(10, -8)
	iterations := 0

	var meanHyperparams [len(hyperparams)]float64
	for i := range meanHyperparams {
		meanHyperparams[i] = (lower[i] + upper[i]) / 2.0
	}

	covariance := make([][]float64, len(hyperparams))
	for i := range covariance {
		covariance[i] = make([]float64, len(hyperparams))
	}
	minLower := lower[0]
	maxUpper := upper[0]

	for _, v := range lower {
		if v < minLower {
			minLower = v
		}
	}
	for _, v := range upper {
		if v > maxUpper {
			maxUpper = v
		}
	}

	for i := range covariance {
		for j := range covariance[i] {
			if i == j {
				covariance[i][j] = math.Pow(maxUpper-minLower, 2) + e
			} else {
				covariance[i][j] = e
			}
		}
	}

	covariance = nearestPD(covariance) // write code for that

	// to create MVD you need to use gonum NewNormal

	fmt.Println(numTimesteps, numRuns, hyperparams, lower, upper, discreteHyperparamsIndices, discreteMidRanges, discreteRanges, numSamples, numElite, e, iterations, meanHyperparams, covariance)

	Matrix := mat.NewDense(len(covariance), len(covariance), nil)
	for i := range covariance {
		for j := range covariance[i] {
			Matrix.Set(i, j, covariance[i][j])
		}
	}
	fmt.Println(covariance)
	fmt.Println(Matrix)

	var svd mat.SVD
	//S := svd.Values(Matrix)
	//V := svd.VTo(Matrix)

	fmt.Println("---------")
	fmt.Println(svd.VTo(Matrix))
	//fmt.Println(S)
	//fmt.Println(V)

	for {
		agentSettings := agent.DefaultESarsaSettings()
		// Do CEM stuff to change settings and SEED

		ag := &agent.ESarsa{Debug: debug}
		ag.InitializeWithSettings(agentSettings, lockweight.LockWeight{})

		env := &environment.Cartpole{Debug: debug}
		env.InitializeWithSettings(environment.CartpoleSettings{Seed: int64(0)}) // TODO change seed

		data, err := logger.NewData(debug, logger.DataConfig{
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			BasePath:                "",
			FileSuffix:              "",
		})
		panicIfError(err, "Couldn't create logger.Data")

		expConf := config.Experiment{
			MaxEpisodes:             1,
			MaxSteps:                0,
			DebugInterval:           0,
			DataPath:                "",
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			MaxCPUs:                 *cpus,
		}
		exp, err := experiment.New(ag, env, expConf, debug, data)
		panicIfError(err, "Couldn't create experiment")

		exp.Run()

		//fmt.Println(data.NumberOfEpisodes())
		break
	}
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}

func nearestPD(A [][]float64) [][]float64 {
	//B := (A + transpose(A)) / 2.0

	//A2 := (B + H) / 2.0
	//A3 := (A2 + transpose(A2)) / 2.0
	return A
}

func isPD(matrix [][]float64) bool {
	boolean := true
	return boolean
}

func transpose(matrix [][]float64) [][]float64 {
	transposeMatrix := make([][]float64, len(matrix))
	for i := range transposeMatrix {
		transposeMatrix[i] = make([]float64, len(matrix))
	}

	for i := range matrix {
		for j := range matrix[i] {
			transposeMatrix[i][j] = matrix[j][i]
		}

	}
	return transposeMatrix
}
