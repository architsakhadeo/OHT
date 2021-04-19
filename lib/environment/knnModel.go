package environment

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/stellentus/cartpoles/lib/representation"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"gonum.org/v1/gonum/mat"
	"path"

	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/util/convformat"
	"github.com/stellentus/cartpoles/lib/util/random"
	transModel "github.com/stellentus/cartpoles/lib/util/transkdtree"
	tpo "github.com/stellentus/cartpoles/lib/util/type-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type EpStartFunc func() rlglue.State
type ChooseNeighborFunc func([]float64, [][]float64, [][]float64, []float64, []float64, []float64) ([]float64, []float64, float64, float64, float64)

type knnSettings struct {
	DataLog      string  `json:"datalog"`
	TrueStartLog string  `json:"true-start-log"`
	Seed         int64   `json:"seed"`
	TotalLogs    uint    `json:"total-logs"`
	Neighbor_num int     `json:"neighbor-num"`
	EnsembleSeed int     `json:"ensemble-seed"`
	DropPerc     float64 `json:"drop-percent"`
	//Timeout      int 	 `json:"timeout"`
	PickStartS  string  `json:"pick-start-state"`
	PickNext    string  `json:"pick-next"`
	NoisyS      float64 `json:"state-noise"`
	ShapeReward bool    `json:"shape-reward"`
	ExtraRisk   float64 `json:"extra-risk"`
}

type repSettings struct {
	TrainStep		int 	`json:"rep-train-num-step"`
	TrainBeta		float64 `json:"rep-train-beta"`
	TrainDelta		float64 `json:"rep-train-delta"`
	TrainLambda		float64 `json:"rep-train-lambda"`
	TrainTrajLen	int 	`json:"rep-train-traj-len"`
	TrainBatch		int 	`json:"rep-train-batch"`
	LearnRate		float64 `json:"rep-train-learning-rate"`
	TrainHiddenLy	[]int 	`json:"rep-hidden"`

	RepLen			int 	`json:"rep-dim"`
	RepName			string 	`json:"rep-name"`
}

type KnnModelEnv struct {
	logger.Debug
	knnSettings
	repSettings
	state           rlglue.State
	rng             *rand.Rand

	offlineDataRep     [][]float64
	offlineDataObs     [][]float64
	offlineStarts   []int
	offlineTermns   []int
	trueDataRep        [][]float64
	trueDataObs        [][]float64
	trueStarts      []int
	trueTermns      []int
	offlineModel    transModel.TransTrees
	terminsTree     transModel.TransTrees
	stateDim        int
	NumberOfActions int
	stateRange      []float64
	//neighbor_prob   []float64
	rewardBound []float64
	stateBound  [][]float64
	PickStartFunc EpStartFunc
	PickNextFunc  ChooseNeighborFunc

	maxSDist float64
	DebugArr [][]float64
	TempCount			int

	VisitCount	map[string]int
}

func init() {
	Add("knnModel", NewKnnModelEnv)
}

func NewKnnModelEnv(logger logger.Debug) (rlglue.Environment, error) {
	return &KnnModelEnv{Debug: logger}, nil
}

func (env *KnnModelEnv) SettingFromLog(paramLog string) {
	// Get state dimension
	txt, err := os.Open(paramLog)
	if err != nil {
		panic(err)
	}
	defer txt.Close()
	scanner := bufio.NewScanner(txt)
	var line string
	var spl []string
	var stateDim int
	var numAction int
	var stateRange []float64
	for scanner.Scan() {
		line = scanner.Text()
		spl = strings.Split(line, "=")
		if spl[0] == "stateDimension" { //stateDimension
			stateDim, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "numberOfActions" {
			numAction, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "stateRange" {
			stateRange = convformat.ListStr2Float(spl[1], ",")
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	env.stateDim = stateDim
	env.NumberOfActions = numAction
	env.stateRange = stateRange
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *KnnModelEnv) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.knnSettings)
	if err != nil {
		err = errors.New("environment.knnModel settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	err = json.Unmarshal(attr, &env.repSettings)
	if err != nil {
		err = errors.New("environment.knnModel settings error: " + err.Error())
		env.Message("err", err)
		return err
	}

	env.knnSettings.Seed += int64(run / env.knnSettings.TotalLogs)
	// For CEM, use env.knnSettings.Seed += int64(run)

	env.rng = rand.New(rand.NewSource(env.knnSettings.Seed)) // Create a new rand source for reproducibility
	env.TempCount = 0

	env.Message("environment.knnModel settings", fmt.Sprintf("%+v", env.knnSettings))

	folder := env.knnSettings.DataLog
	var trueStartFolder string
	if env.knnSettings.TrueStartLog != "" {
		trueStartFolder = env.knnSettings.TrueStartLog
	} else {
		trueStartFolder = folder
	}
	//traceLog := folder + "/traces-" + strconv.Itoa(int(run)) + ".csv"
	traceLog := folder + "/traces-" + strconv.Itoa(int(run%env.knnSettings.TotalLogs)) + ".csv"
	trueStartLog := trueStartFolder + "/traces-" + strconv.Itoa(int(run%env.knnSettings.TotalLogs)) + ".csv"
	//fmt.Println(traceLog)
	//fmt.Println(trueStartLog)

	env.Message("KNN data log", traceLog, "\n")
	env.Message("KNN starts log", trueStartLog, "\n")
	paramLog := folder + "/log_json.txt"
	env.SettingFromLog(paramLog)
	env.state = make(rlglue.State, env.stateDim)

	//env.offlineData = allTrans
	env.offlineDataObs, env.offlineDataRep = env.LoadData(traceLog)
	env.trueDataObs, env.trueDataRep = env.LoadData(trueStartLog)
	//env.offlineStarts, env.offlineTermns = env.SearchOfflineStart(offlineStartsData)
	env.trueStarts, env.trueTermns = env.SearchOfflineStart(env.trueDataObs)

	env.VisitCount = make(map[string]int)

	if env.knnSettings.ExtraRisk > 0 {
		numIdx := int(float64(len(env.offlineDataObs)) * env.knnSettings.ExtraRisk)
		for i := 0; i < numIdx; i++ {
			rndIdx := env.rng.Int() % len(env.offlineDataObs)
			env.offlineDataObs[rndIdx][env.stateDim*2+1] = env.rewardBound[0]
			env.offlineDataObs[rndIdx][env.stateDim*2+2] = 1
			env.offlineDataRep[rndIdx][env.stateDim*2+1] = env.rewardBound[0]
			env.offlineDataRep[rndIdx][env.stateDim*2+2] = 1
		}
	}
	env.offlineModel = transModel.New(env.NumberOfActions, env.stateDim)
	env.offlineModel.BuildTree(env.offlineDataRep, "current")

	if env.knnSettings.PickStartS == "random-init" { // default setting
		env.PickStartFunc = env.randomizeInitState
	} else if env.knnSettings.PickStartS == "furthest" {
		env.PickStartFunc = env.furthestState
	} else if env.knnSettings.PickStartS == "random-all" {
		env.PickStartFunc = env.randomizeState
	} else {
		env.PickStartFunc = env.randomizeInitState
	}

	if env.knnSettings.PickNext == "furthest" {
		env.PickNextFunc = env.FurtherNext
	} else if env.knnSettings.PickNext == "pessimistic" {
		env.PickNextFunc = env.LowRwdNext
	} else { // default setting
		env.PickNextFunc = env.CloserNeighbor
	}

	if env.ShapeReward {
		var trans2term [][]float64
		for i := 0; i < len(env.trueTermns); i++ {
			trans2term = append(trans2term, env.offlineDataObs[env.trueTermns[i]])
		}
		env.terminsTree = transModel.New(env.NumberOfActions, env.stateDim)
		env.terminsTree.BuildTree(trans2term, "next")
	}
	return nil
}

func (env *KnnModelEnv) LoadData(filename string) ([][]float64, [][]float64) {
	// Get offline data
	csvFile, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	allTransStr, err := csv.NewReader(csvFile).ReadAll()
	csvFile.Close()
	if err != nil {
		log.Fatal(err)
	}

	allTransObs := make([][]float64, len(allTransStr)-1) // transitions, using environment state
	rewards := make([]float64, len(allTransStr)-1)
	allStates := make([][]float64, len(allTransStr)-1) // current states
	allNextStates := make([][]float64, len(allTransStr)-1) // next states
	allTermin := make([]float64, len(allTransStr)-1)
	for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
		trans := allTransStr[i]
		row := make([]float64, env.stateDim*2+3)
		for j, num := range trans {
			if j == 0 { // next state
				num = num[1 : len(num)-1] // remove square brackets
				copy(row[env.stateDim+1:env.stateDim*2+1], convformat.ListStr2Float(num, " "))

				allNextStates[i-1] = make([]float64, env.stateDim)
				copy(allNextStates[i-1], row[env.stateDim+1 : env.stateDim*2+1])

			} else if j == 1 { // current state
				num = num[1 : len(num)-1]
				copy(row[:env.stateDim], convformat.ListStr2Float(num, " "))

				allStates[i-1] = make([]float64, env.stateDim)
				copy(allStates[i-1], row[:env.stateDim])

			} else if j == 2 { // action
				row[env.stateDim], _ = strconv.ParseFloat(num, 64)

			} else if j == 3 { //reward
				row[env.stateDim*2+1], _ = strconv.ParseFloat(num, 64)
				rewards[i-1] = row[env.stateDim*2+1]

			} else if j == 4 { //termination
				row[env.stateDim*2+2], _ = strconv.ParseFloat(num, 64)
				allTermin[i-1], _ = strconv.ParseFloat(num, 64)
			}
		}
		allTransObs[i-1] = row
	}

	var allNextRep [][]float64
	var allRep [][]float64
	if env.repSettings.RepName == "Laplace" {
		repModel := representation.NewLaplace()
		repModel.Initialize(int(env.knnSettings.Seed), env.repSettings.TrainStep, env.repSettings.TrainBeta, env.repSettings.TrainDelta,
			env.repSettings.TrainLambda, env.repSettings.TrainTrajLen, env.repSettings.TrainBatch, env.repSettings.LearnRate,
			env.repSettings.TrainHiddenLy, allStates, allTermin, env.stateDim, env.repSettings.RepLen)
		repFunc := repModel.Train()
		allNextRep = repFunc.Predict(allNextStates)
		allRep = repFunc.Predict(allStates)
	} else {
		env.repSettings.RepLen = env.stateDim
		allNextRep = allNextStates
		allRep = allStates
	}

	allTransRep := make([][]float64, len(allTransStr)-1)
	for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
		trans := allTransStr[i]
		row := make([]float64, env.repSettings.RepLen*2+3+1) // The last bit is the index
		for j, num := range trans {
			if j == 0 { // next state
				copy(row[env.repSettings.RepLen+1:env.repSettings.RepLen*2+1], allNextRep[i-1])
			} else if j == 1 { // current state
				copy(row[:env.repSettings.RepLen], allRep[i-1])
			} else if j == 2 { // action
				row[env.repSettings.RepLen], _ = strconv.ParseFloat(num, 64)
			} else if j == 3 { //reward
				row[env.repSettings.RepLen*2+1], _ = strconv.ParseFloat(num, 64)
				//rewards[i-1] = row[env.repSettings.RepLen*2+1]
			} else if j == 4 { //termination
				row[env.repSettings.RepLen*2+2], _ = strconv.ParseFloat(num, 64)
			}
		}
		row[env.repSettings.RepLen*2+3] = float64(i-1) // index
		allTransRep[i-1] = row
		//fmt.Println(allTransRep[i-1])
		//fmt.Println(allTransObs[i-1], "\n")
	}

	env.rewardBound = make([]float64, 2)
	env.rewardBound[0], _ = ao.ArrayMin(rewards)
	env.rewardBound[1], _ = ao.ArrayMax(rewards)
	env.stateBound = make([][]float64, 2)
	for i := 0; i < len(allStates[0]); i++ {
		mn, _ := ao.ColumnMin(allStates, i)
		mx, _ := ao.ColumnMax(allStates, i)
		env.stateBound[0] = append(env.stateBound[0], mn)
		env.stateBound[1] = append(env.stateBound[1], mx)
	}
	env.maxSDist = env.Distance(env.stateBound[0], env.stateBound[1])

	if env.NoisyS != 0 {
		for i := 0; i < len(allTransRep); i++ {
			temp := env.AddStateNoise(allTransObs[i][:env.stateDim], env.stateBound)
			copy(allTransObs[i][:env.stateDim], temp)
			temp = env.AddStateNoise(allTransObs[i][env.stateDim+1:env.stateDim*2+1], env.stateBound)
			copy(allTransObs[i][env.stateDim+1:env.stateDim*2+1], temp)
		}
	}

	var allTransRepKeep [][]float64
	var allTransObsKeep [][]float64
	//if env.EnsembleSeed != 0 {
	//	tempRnd := rand.New(rand.NewSource(int64(env.EnsembleSeed)))
	if env.DropPerc != 0 {
		//fmt.Println("HERE", env.DropPerc)
		filteredLen := int(float64(len(allTransRep)) * (1 - env.DropPerc))
		filteredIdx := env.rng.Perm(len(allTransRep))[:filteredLen]
		//fmt.Println("HERE", filteredIdx[:5])
		allTransRepKeep = make([][]float64, filteredLen)
		allTransObsKeep = make([][]float64, filteredLen)
		for i := 0; i < filteredLen; i++ {
			allTransRepKeep[i] = allTransRep[filteredIdx[i]]
			allTransObsKeep[i] = allTransObs[filteredIdx[i]]
		}
	} else {
		allTransRepKeep = allTransRep
		allTransObsKeep = allTransObs
	}
	return allTransObsKeep, allTransRepKeep
}

func (env *KnnModelEnv) SearchOfflineStart(allTrans [][]float64) ([]int, []int) {
	starts := []int{0}
	termins := []int{}
	for i := 0; i < len(allTrans)-1; i++ { // not include the end of run
		if allTrans[i][len(allTrans[i])-1] == 1 {
			starts = append(starts, i+1)
			termins = append(termins, i)
			//fmt.Println(allTrans[i], i, termins)
		}
	}
	return starts, termins
}

func (env *KnnModelEnv) randomizeInitState() rlglue.State {
	randIdx := env.rng.Intn(len(env.trueStarts))
	state := env.trueDataObs[env.trueStarts[randIdx]][:env.stateDim]
	return state
}

///* For debugging only */
//func (env *KnnModelEnv) cheatingChoice() rlglue.State {
//	randIdx := env.rng.Intn(len(env.trueStarts))
//	state := env.offlineData[env.trueStarts[randIdx]-1]
//	fmt.Println("terminal ", state)
//	state = state[:env.stateDim]
//	return state
//}

func (env *KnnModelEnv) randomizeState() rlglue.State {
	randIdx := env.rng.Intn(len(env.offlineDataObs))
	state := env.offlineDataObs[randIdx][:env.stateDim]
	return state
}

func (env *KnnModelEnv) furthestState() rlglue.State {
	totalSize := 0
	for act := 0; act < env.NumberOfActions; act++ {
		totalSize += env.offlineModel.TreeSize(act)
	}
	totalStates := make([][]float64, totalSize)
	totalDistance := make([]float64, totalSize)
	idx := 0
	size := 0
	for act := 0; act < env.NumberOfActions; act++ {
		size = env.offlineModel.TreeSize(act)
		_, _, _, _, distances, repIdxs := env.offlineModel.SearchTree(env.state, act, size)
		states := make([][]float64, len(repIdxs))
		for j, id := range repIdxs {
			copy(states[j], env.offlineDataObs[id][:env.stateDim])
		}
		copy(totalStates[idx:idx+size], states)
		copy(totalDistance[idx:idx+size], distances)
		idx += size
	}
	//fmt.Printf("\n%.2f \n", env.state)

	sum := 0.0
	for i := 0; i < len(totalDistance); i++ {
		sum += totalDistance[i] + math.Pow(10, -6)
	}
	pdf := make([]float64, len(totalDistance))
	for i := 0; i < len(totalDistance); i++ {
		pdf[i] = (totalDistance[i] + math.Pow(10, -6)) / sum
	}
	normalizedPdf := make([]float64, len(totalDistance))
	normalizedSum := 0.0
	for i := 0; i < len(totalDistance); i++ {
		normalizedSum += pdf[i]
	}
	for i := 0; i < len(totalDistance); i++ {
		normalizedPdf[i] = (pdf[i] / normalizedSum)
	}

	//prob := make([]float64, len(totalDistance))
	//temp1 := 0.0
	//for i := 0; i < len(totalDistance); i++ {
	//	prob[i] = temp1 + normalizedPdf[i]
	//	temp1 = prob[i]
	//}
	//chosen := random.FreqSample(prob)
	chosen := random.FreqSample(normalizedPdf)
	state := totalStates[chosen]
	//fmt.Printf("%.2f \n", totalDistance[chosen])
	return state
}

// Start returns an initial observation.
func (env *KnnModelEnv) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	//env.Count = 0
	env.state = env.PickStartFunc()
	//env.state = env.randomizeInitState()
	//env.state = env.randomizeState()
	//env.state = env.furthestState()
	//env.state = env.cheatingChoice()
	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)

	key := ao.FloatAryToString(env.state, " ")
	env.VisitCount[key] += 1
	env.TempCount += 1
	//fmt.Println("---", key, env.VisitCount[key])

	//fmt.Println("Start", env.state)
	return state_copy, ""
}

func (env *KnnModelEnv) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	//fmt.Println("---------------------")
	actInt, _ := tpo.GetInt(act)

	if env.offlineModel.TreeSize(actInt) == 0 {
		log.Printf("Warning: There is no data for action %d, terminating the episode\n", act)
		startReturn, _ := env.Start(randomizeStartStateCondition)
		return startReturn, env.rewardBound[0], true, ""
	}
	//fmt.Println("Before target:", env.state)
	//fmt.Println(env.offlineModel.SearchTree(env.state, actInt, env.Neighbor_num))
	_, _, rewards, terminals, distances, repIdxs := env.offlineModel.SearchTree(env.state, actInt, env.Neighbor_num)
	states := make([][]float64, len(repIdxs))
	nextStates := make([][]float64, len(repIdxs))
	for j, id := range repIdxs {
		states[j] = make([]float64, env.stateDim)
		nextStates[j] = make([]float64, env.stateDim)
		copy(states[j], env.offlineDataObs[id][:env.stateDim])
		copy(nextStates[j], env.offlineDataObs[id][env.stateDim+1 : env.stateDim*2+1])
	}

	temp, nextState, reward, terminal, _ := env.PickNextFunc(env.state, states, nextStates, rewards, terminals, distances)

	env.state = nextState

	//idx := ao.Search2D(env.state, env.DebugArr)
	//if idx > -1 {
	//	fmt.Printf("%d, %d, %.2f, %d, \n", act, idx, env.DebugArr[idx], terminals[chosen])
	//} else {
	//	fmt.Println(act, idx, terminals[chosen])
	//}
	//env.DebugArr = append(env.DebugArr, env.state)

	var done bool
	//if terminals[chosen] == 0 {
	if terminal == 0 {
		done = false
	} else {
		done = true
		fmt.Println("Terminal state", temp)
	}
	key := ao.FloatAryToString(env.state, " ")
	env.VisitCount[key] += 1
	//fmt.Println("---", key, env.VisitCount[key])

	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)

	if env.ShapeReward {
		var totalRwd []float64
		var totalDistance []float64
		for act := 0; act < env.NumberOfActions; act++ {
			_, _, tR, _, dist, _ := env.terminsTree.SearchTree(env.state, act, 1)
			if tR != nil {
				totalRwd = append(totalRwd, tR[0])
				totalDistance = append(totalDistance, dist[0])
			}
		}
		if len(totalRwd) != 0 {
			minD, idx := ao.ArrayMin(totalDistance)
			termR := totalRwd[idx]
			scale := 1.0 - minD/env.maxSDist
			reward += scale * (termR - reward)
			//fmt.Println(reward, scale, minD, env.maxSDist)
		}
	}

	env.TempCount += 1
	//fmt.Println(env.TempCount)
	//if env.TempCount == 49999 {
	//	env.PrintVisitLog()
	//}
	return state_copy, reward, done, ""
}

//GetAttributes returns attributes for this environment.
func (env *KnnModelEnv) GetAttributes() rlglue.Attributes {
	// Add elements to attributes.
	attributes := struct {
		NumAction  int       `json:"numberOfActions"`
		StateDim   int       `json:"stateDimension"`
		StateRange []float64 `json:"stateRange"`
	}{
		env.NumberOfActions,
		env.stateDim,
		env.stateRange,
	}
	attr, err := json.Marshal(&attributes)
	if err != nil {
		env.Message("err", "environment.knnModel could not Marshal its JSON attributes: "+err.Error())
	}
	return attr
}

func (env *KnnModelEnv) Distance(state1 rlglue.State, state2 rlglue.State) float64 {
	var distance float64
	var squareddistance float64
	squareddistance = 0.0
	for i := 0; i < len(state1); i++ {
		squareddistance += math.Pow(state1[i]-state2[i], 2)
	}
	distance = math.Pow(squareddistance, 0.5)
	return distance
}

func (env *KnnModelEnv) AddStateNoise(data []float64, bound [][]float64) []float64 {
	for j := 0; j < len(data); j++ {
		data[j] += env.rng.Float64() * env.NoisyS * (bound[1][j] - bound[0][j])
	}
	return data
}

func (env *KnnModelEnv) CloserNeighbor(currentS []float64, states, nextStates [][]float64, rewards, terminals, distances []float64) ([]float64, []float64, float64, float64, float64) {
	sum := 0.0
	for i := 0; i < len(distances); i++ {
		sum += distances[i] + math.Pow(10, -6)
	}
	pdf := make([]float64, len(distances))
	for i := 0; i < len(distances); i++ {
		pdf[i] = 1.0 - ((distances[i] + math.Pow(10, -6)) / sum)
	}
	normalizedPdf := make([]float64, len(distances))
	normalizedSum := 0.0
	for i := 0; i < len(distances); i++ {
		normalizedSum += pdf[i]
	}
	for i := 0; i < len(distances); i++ {
		normalizedPdf[i] = (pdf[i] / normalizedSum)
	}
	//neighbor_prob := make([]float64, len(distances))
	//temp1 := 0.0
	//for i := 0; i < len(distances); i++ {
	//	neighbor_prob[i] = temp1 + normalizedPdf[i]
	//	temp1 = neighbor_prob[i]
	//}
	//chosen := random.FreqSample(neighbor_prob)
	chosen := random.FreqSample(normalizedPdf)

	state := states[chosen]
	nextState := nextStates[chosen]
	reward := rewards[chosen]
	terminal := terminals[chosen]
	distance := distances[chosen]
	//fmt.Println("After:", state, nextState, reward, terminal, distance, distances)
	return state, nextState, reward, terminal, distance
}

func (env *KnnModelEnv) FurtherNext(currentS []float64, states, nextStates [][]float64, rewards, terminals, distances []float64) ([]float64, []float64, float64, float64, float64) {
	//fmt.Println(currentS, nextStates)
	vectorX := mat.NewDense(len(currentS), 1, currentS)

	dists := make([]float64, len(nextStates))
	for i := 0; i < len(nextStates); i++ {
		vectorY := mat.NewDense(len(nextStates[i]), 1, nextStates[i])
		var temp mat.Dense
		temp.Sub(vectorX, vectorY)
		dists[i] = mat.Norm(&temp, 2)
	}
	//_, chosen := ao.ArrayMax(dists)

	sum := 0.0
	for i := 0; i < len(dists); i++ {
		sum += dists[i] + math.Pow(10, -6)
	}
	pdf := make([]float64, len(dists))
	for i := 0; i < len(dists); i++ {
		//pdf[i] = 1.0 - ((dists[i] + math.Pow(10, -6)) / sum)
		pdf[i] = (dists[i] + math.Pow(10, -6)) / sum
	}
	normalizedPdf := make([]float64, len(dists))
	normalizedSum := 0.0
	for i := 0; i < len(dists); i++ {
		normalizedSum += pdf[i]
	}
	for i := 0; i < len(dists); i++ {
		normalizedPdf[i] = (pdf[i] / normalizedSum)
	}
	//neighbor_prob := make([]float64, len(dists))
	//temp1 := 0.0
	//for i := 0; i < len(dists); i++ {
	//	neighbor_prob[i] = temp1 + normalizedPdf[i]
	//	temp1 = neighbor_prob[i]
	//}
	chosen := random.FreqSample(normalizedPdf)

	state := states[chosen]
	nextState := nextStates[chosen]
	reward := rewards[chosen]
	terminal := terminals[chosen]
	distance := distances[chosen]
	return state, nextState, reward, terminal, distance
}

func (env *KnnModelEnv) LowRwdNext(currentS []float64, states, nextStates [][]float64, rewards, terminals, distances []float64) ([]float64, []float64, float64, float64, float64) {

	rwds := make([]float64, len(rewards))
	minRwd, _ := ao.ArrayMin(rewards)
	for i := 0; i < len(rewards); i++ {
		rwds[i] = rewards[i] + minRwd
	}

	sum := 0.0
	for i := 0; i < len(rwds); i++ {
		sum += rwds[i] + math.Pow(10, -6)
	}
	pdf := make([]float64, len(rwds))
	for i := 0; i < len(rwds); i++ {
		pdf[i] = 1.0 - ((rwds[i] + math.Pow(10, -6)) / sum)
	}
	normalizedPdf := make([]float64, len(rwds))
	normalizedSum := 0.0
	for i := 0; i < len(rwds); i++ {
		normalizedSum += pdf[i]
	}
	for i := 0; i < len(rwds); i++ {
		normalizedPdf[i] = (pdf[i] / normalizedSum)
	}
	//neighbor_prob := make([]float64, len(rwds))
	//temp1 := 0.0
	//for i := 0; i < len(rwds); i++ {
	//	neighbor_prob[i] = temp1 + normalizedPdf[i]
	//	temp1 = neighbor_prob[i]
	//}
	chosen := random.FreqSample(normalizedPdf)

	state := states[chosen]
	nextState := nextStates[chosen]
	reward := rewards[chosen]
	terminal := terminals[chosen]
	distance := distances[chosen]
	//fmt.Println("===", rewards, reward)
	return state, nextState, reward, terminal, distance
}


func (env *KnnModelEnv) GetInfo(info string, value float64) interface{} {
	if info == "visitCount" {
		return env.LogVisitCount()
	} else {
		return nil
	}
}
func (env *KnnModelEnv) LogVisitCount() error {
	file, err := os.Create(path.Join("plot/temp/", "visits-"+strconv.Itoa(int(env.knnSettings.Seed))+".csv"))
	if err != nil {
		return err
	}
	defer file.Close()

	// Write header row
	_, err = file.WriteString("states,visits\n")
	if err != nil {
		return err
	}
	// Write totals
	//_, err = file.WriteString(fmt.Sprintf("%f,%d\n", lg.totalReward, lg.totalEpisodes))
	//if err != nil {
	//	return err
	//}
	for i:=0; i<len(env.offlineDataObs); i++ {
		key := ao.FloatAryToString(env.offlineDataObs[i][env.stateDim+1:env.stateDim*2+1], " ")
		_, err := file.WriteString(fmt.Sprintf("%v,%d\n", key, env.VisitCount[key]))
		if err != nil {
			return err
		}
	}
	//for state, count := range env.VisitCount {
	//	fmt.Println(state, count)
	//	_, err := file.WriteString(fmt.Sprintf("%s,%d\n", state, count))
	//}

	fmt.Println("Save log in", path.Join("plot/temp/", "visits-"+string(env.knnSettings.Seed)+".csv"))
	return err
}
