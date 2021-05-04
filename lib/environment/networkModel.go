package environment

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	transModel "github.com/stellentus/cartpoles/lib/util/transModel/transnetwork"
	tpo "github.com/stellentus/cartpoles/lib/util/type-opr"

	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/util/convformat"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type EpStartFuncNet func() rlglue.State

type networkSettings struct {
	DataLog      string  `json:"datalog"`
	TrueStartLog string  `json:"true-start-log"`
	Seed         int64   `json:"seed"`
	TotalLogs    uint    `json:"total-logs"`
	EnsembleSeed int     `json:"ensemble-seed"`
	DropPerc     float64 `json:"drop-percent"`
	//Timeout      int 	 `json:"timeout"`
	PickStartS string  `json:"pick-start-state"`

	TrainEpoch	int 	`json:"train-epoch"`
	TrainBatchSize	int `json:"train-batch"`
	HiddenLayer []int 	`json:"train-hidden-layer"`
	TrainLearningRate float64 	`json:"train-learning-rate"`

	IsTest	bool 	`json:"is-test"`

	ScaleInput bool 	`json:"scale-input"`
	ClipPrediction bool `json:"clip-prediction"`
}

type networkModelEnv struct {
	logger.Debug
	networkSettings
	//repSettings
	Trained bool
	//repFunc network.Network

	//rep   []float64
	state rlglue.State
	rng   *rand.Rand

	//offlineDataRep  [][]float64
	offlineDataObs [][]float64
	offlineStarts  []int
	offlineTermns  []int
	//trueDataRep     [][]float64
	trueDataObs [][]float64
	trueStarts  []int
	trueTermns  []int

	offlineModel    *transModel.TransNetwork

	stateDim        int
	NumberOfActions int
	stateRange      []float64
	rewardBound     [][]float64
	stateBound      [][]float64
	PickStartFunc   EpStartFuncNet

	DebugArr  [][]float64
}

func init() {
	Add("networkModel", NewnetworkModelEnv)
}

func NewnetworkModelEnv(logger logger.Debug) (rlglue.Environment, error) {
	return &networkModelEnv{Debug: logger}, nil
}

func (env *networkModelEnv) SettingFromLog(paramLog string) {
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
func (env *networkModelEnv) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.networkSettings)
	if err != nil {
		err = errors.New("environment.networkModel settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	//err = json.Unmarshal(attr, &env.repSettings)
	//if err != nil {
	//	err = errors.New("environment.networkModel settings error: " + err.Error())
	//	env.Message("err", err)
	//	return err
	//}
	//if env.repSettings.RepName == "Laplace" {
	//	env.ScaleInput = false // Do not scale input if using representation
	//}

	env.networkSettings.Seed += int64(run / env.networkSettings.TotalLogs)
	// For CEM, use env.networkSettings.Seed += int64(run)

	env.rng = rand.New(rand.NewSource(env.networkSettings.Seed)) // Create a new rand source for reproducibility

	env.Message("environment.networkModel settings", fmt.Sprintf("%+v", env.networkSettings))

	folder := env.networkSettings.DataLog
	var trueStartFolder string
	if env.networkSettings.TrueStartLog != "" {
		trueStartFolder = env.networkSettings.TrueStartLog
	} else {
		trueStartFolder = folder
	}
	traceLog := folder + "/traces-" + strconv.Itoa(int(run%env.networkSettings.TotalLogs)) + ".csv"
	trueStartLog := trueStartFolder + "/traces-" + strconv.Itoa(int(run%env.networkSettings.TotalLogs)) + ".csv"

	env.Message("network data log", traceLog, "\n")
	env.Message("network starts log", trueStartLog, "\n")
	paramLog := folder + "/log_json.txt"
	env.SettingFromLog(paramLog)
	env.state = make(rlglue.State, env.stateDim)

	//env.offlineDataObs, env.offlineDataRep = env.LoadData(traceLog)
	env.offlineDataObs = env.LoadData(traceLog)
	fmt.Println("Offline Data Loaded")
	env.trueDataObs = env.LoadData(trueStartLog)
	fmt.Println("True Data Loaded")

	env.trueStarts, env.trueTermns = env.SearchOfflineStart(env.trueDataObs)

	env.offlineModel = transModel.New()
	var trainingData [][]float64
	if env.ScaleInput {
		trainingData = env.ScaleTrans(env.offlineDataObs, env.stateBound, env.rewardBound)
	//} else if env.repSettings.RepName == "Laplace" {
	//	trainingData = env.offlineDataRep
	} else {
		trainingData = env.offlineDataObs
	}
	env.offlineModel.Initialize(env.networkSettings.Seed, trainingData, env.networkSettings.TrainEpoch, env.networkSettings.TrainBatchSize,
		env.networkSettings.TrainLearningRate, env.networkSettings.HiddenLayer, env.stateDim, env.NumberOfActions)
	if !env.networkSettings.IsTest {
		env.offlineModel.Train()
	} else {
		env.offlineModel.CrossValidation()
		return nil
	}

	if env.networkSettings.PickStartS == "random-init" { // default setting
		env.PickStartFunc = env.randomizeInitState
	} else {
		env.PickStartFunc = env.randomizeInitState
	}

	return nil
}

func (env *networkModelEnv) LoadData(filename string) [][]float64 {
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
	allStates := make([][]float64, len(allTransStr)-1)     // current states
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
				copy(allNextStates[i-1], row[env.stateDim+1:env.stateDim*2+1])

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

	//var allNextRep [][]float64
	//var allRep [][]float64
	//if env.repSettings.RepName == "Laplace" {
	//	if !env.Trained {
	//		repModel := representation.NewLaplace()
	//		repModel.Initialize(int(env.networkSettings.Seed), env.repSettings.TrainStep, env.repSettings.TrainBeta, env.repSettings.TrainDelta,
	//			env.repSettings.TrainLambda, env.repSettings.TrainTrajLen, env.repSettings.TrainBatch, env.repSettings.LearnRate,
	//			env.repSettings.TrainHiddenLy, allStates, allTermin, env.stateDim, env.repSettings.RepLen, env.repSettings.TestForward)
	//		if !env.repSettings.IsTest {
	//			env.repFunc = repModel.Train()
	//		} else {
	//			env.repFunc = repModel.CrossValidation()
	//		}
	//
	//		log.Println("Representation has been trained")
	//		env.Trained = true
	//	}
	//	allNextRep = env.repFunc.Predict(allNextStates)
	//	allRep = env.repFunc.Predict(allStates)
	//	log.Println("Obs to Rep")
	//} else {
	//	env.repSettings.RepLen = env.stateDim
	//	allNextRep = allNextStates
	//	allRep = allStates
	//}
	//
	//allTransRep := make([][]float64, len(allTransStr)-1)
	//for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
	//	trans := allTransStr[i]
	//	row := make([]float64, env.repSettings.RepLen*2+3+1) // The last bit is the index
	//	for j, num := range trans {
	//		if j == 0 { // next state
	//			copy(row[env.repSettings.RepLen+1:env.repSettings.RepLen*2+1], allNextRep[i-1])
	//		} else if j == 1 { // current state
	//			copy(row[:env.repSettings.RepLen], allRep[i-1])
	//		} else if j == 2 { // action
	//			row[env.repSettings.RepLen], _ = strconv.ParseFloat(num, 64)
	//		} else if j == 3 { //reward
	//			row[env.repSettings.RepLen*2+1], _ = strconv.ParseFloat(num, 64)
	//			//rewards[i-1] = row[env.repSettings.RepLen*2+1]
	//		} else if j == 4 { //termination
	//			row[env.repSettings.RepLen*2+2], _ = strconv.ParseFloat(num, 64)
	//		}
	//	}
	//	row[env.repSettings.RepLen*2+3] = float64(i - 1) // index
	//	allTransRep[i-1] = row
	//	//fmt.Println(allTransRep[i-1])
	//	//fmt.Println(allTransObs[i-1], "\n")
	//}

	env.rewardBound = make([][]float64, 2)
	env.rewardBound[0] = make([]float64, 1)
	env.rewardBound[1] = make([]float64, 1)
	env.rewardBound[0][0], _ = ao.ArrayMin(rewards)
	env.rewardBound[1][0], _ = ao.ArrayMax(rewards)
	fmt.Println("Rewards min, max:", env.rewardBound[0], env.rewardBound[1])
	env.stateBound = make([][]float64, 2)
	for i := 0; i < len(allStates[0]); i++ {
		mn, _ := ao.ColumnMin(allStates, i)
		mx, _ := ao.ColumnMax(allStates, i)
		env.stateBound[0] = append(env.stateBound[0], mn)
		env.stateBound[1] = append(env.stateBound[1], mx)
	}
	fmt.Println("States min, max:", env.stateBound[0], env.stateBound[1])

	//var allTransRepKeep [][]float64
	var allTransObsKeep [][]float64
	if env.DropPerc != 0 {
		filteredLen := int(float64(len(allTransObs)) * (1 - env.DropPerc))
		filteredIdx := env.rng.Perm(len(allTransObs))[:filteredLen]
		//allTransRepKeep = make([][]float64, filteredLen)
		allTransObsKeep = make([][]float64, filteredLen)
		for i := 0; i < filteredLen; i++ {
			//allTransRepKeep[i] = allTransRep[filteredIdx[i]]
			allTransObsKeep[i] = allTransObs[filteredIdx[i]]
		}
	} else {
		allTransObsKeep = allTransObs
	}
	//return allTransObsKeep, allTransRepKeep
	return allTransObsKeep
}

func (env *networkModelEnv) SearchOfflineStart(allTrans [][]float64) ([]int, []int) {
	starts := []int{0}
	termins := []int{}
	for i := 0; i < len(allTrans)-1; i++ { // not include the end of run
		if allTrans[i][len(allTrans[i])-1] == 1 {
			starts = append(starts, i+1)
			termins = append(termins, i)
		}
	}
	return starts, termins
}

func (env *networkModelEnv) randomizeInitState() rlglue.State {
	randIdx := env.rng.Intn(len(env.trueStarts))
	state := env.trueDataObs[env.trueStarts[randIdx]][:env.stateDim]
	return state
}

// Start returns an initial observation.
func (env *networkModelEnv) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	env.state = env.PickStartFunc()
	//if env.repSettings.RepName == "Laplace" {
	//	temp := make([][]float64, 1)
	//	temp[0] = make([]float64, len(env.state))
	//	temp[0] = env.state
	//	env.rep = env.repFunc.Predict(temp)[0]
	//} else {
	//	env.rep = env.state
	//}
	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)

	key := ao.FloatAryToString(env.state, " ")
	fmt.Println("Key: ", key)
	return state_copy, ""
}

func (env *networkModelEnv) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	actInt, _ := tpo.GetInt(act)
	if env.ScaleInput {
		env.state, _ = env.Normalizer(env.state, nil, env.stateBound)
	}
	nextState, reward, done := env.offlineModel.PredictSingleTrans(env.state, float64(actInt))
	if env.ScaleInput {
		nextState, _ = env.UnNormalizer(nextState, nil, env.stateBound)
		reward2d := make([]float64, 1)
		reward2d[0] = reward
		reward2d, _ = env.UnNormalizer(reward2d, nil, env.rewardBound)
		reward = reward2d[0]
	}
	if env.ClipPrediction {
		for i:=0; i<env.stateDim; i++{
			temp := math.Max(nextState[i], env.stateBound[0][i])
			temp = math.Min(temp, env.stateBound[1][i])
			nextState[i] = temp
		}
		temp := math.Max(reward, env.rewardBound[0][0])
		temp = math.Min(temp, env.rewardBound[1][0])
		reward = temp
	}
	env.state = nextState
	//if env.repSettings.RepName == "Laplace" {
	//	temp := make([][]float64, 1)
	//	temp[0] = make([]float64, len(env.state))
	//	temp[0] = env.state
	//	env.rep = env.repFunc.Predict(temp)[0]
	//} else {
	//	env.rep = env.state
	//}

	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)
	return state_copy, reward, done, ""
}

//GetAttributes returns attributes for this environment.
func (env *networkModelEnv) GetAttributes() rlglue.Attributes {
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
		env.Message("err", "environment.networkModel could not Marshal its JSON attributes: "+err.Error())
	}
	return attr
}

func (env *networkModelEnv) ScaleTrans(trans [][]float64, stateBound [][]float64, rewardBound [][]float64) ([][]float64) {
	states := ao.Index2d(trans, 0, len(trans), 0, env.stateDim)
	actions := ao.Index2d(trans, 0, len(trans), env.stateDim, env.stateDim+1)
	nextStates := ao.Index2d(trans, 0, len(trans), env.stateDim+1, env.stateDim*2+1)
	rewards := ao.Index2d(trans, 0, len(trans), env.stateDim*2+1, env.stateDim*2+2)
	termins := ao.Index2d(trans, 0, len(trans), env.stateDim*2+2, env.stateDim*2+3)

	_, states = env.Normalizer(nil, states, stateBound)
	_, nextStates = env.Normalizer(nil, nextStates, stateBound)
	_, rewards = env.Normalizer(nil, rewards, rewardBound)
	scaledTrans := ao.Concatenate(ao.Concatenate(ao.Concatenate(ao.Concatenate(states, actions), nextStates), rewards), termins)
	return scaledTrans
}

func (env *networkModelEnv) Normalizer(input []float64, inputs [][]float64, inputBound [][]float64) ([]float64, [][]float64) {
	if input == nil {
		res := make([][]float64, len(inputs))
		dim := len(inputs[0])
		for i:=0; i<len(inputs); i++ {
			res[i] = make([]float64, len(inputs[0]))
			for j:=0; j<dim; j++ {
				res[i][j] = (inputs[i][j] - inputBound[0][j]) / (inputBound[1][j] - inputBound[0][j]) * 2 - 1
			}
		}
		return nil, res
	} else if inputs == nil {
		res := make([]float64, len(input))
		dim := len(input)
		for j:=0; j<dim; j++ {
			res[j] = (input[j] - inputBound[0][j]) / (inputBound[1][j] - inputBound[0][j]) * 2 - 1
		}
		return res, nil
	} else {
		return nil, nil
	}
}

func (env *networkModelEnv) UnNormalizer(input []float64, inputs [][]float64, inputBound [][]float64) ([]float64, [][]float64) {
	if input == nil {
		res := make([][]float64, len(inputs))
		dim := len(inputs[0])
		for i:=0; i<len(inputs); i++ {
			res[i] = make([]float64, len(inputs[0]))
			for j:=0; j<dim; j++ {
				res[i][j] = (inputs[i][j] + 1) / 2.0 * (inputBound[1][j] - inputBound[0][j]) + inputBound[0][j]
			}
		}
		return nil, res
	} else if inputs == nil {
		res := make([]float64, len(input))
		dim := len(input)
		for j:=0; j<dim; j++ {
			res[j] = (input[j] + 1) / 2.0 * (inputBound[1][j] - inputBound[0][j]) + inputBound[0][j]
		}
		return res, nil
	} else {
		return nil, nil
	}
}


func (env *networkModelEnv) GetInfo(info string, value float64) interface{} {
	return nil
}
