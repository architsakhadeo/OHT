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
	NoisyS     float64 `json:"state-noise"`

	TrainEpoch	int 	`json:"train-epoch"`
	TrainBatchSize	int `json:"train-batch"`
	HiddenLayer []int 	`json:"train-hidden-layer"`
	TrainLearningRate float64 	`json:"train-learning-rate"`
}

type networkModelEnv struct {
	logger.Debug
	networkSettings
	Trained bool

	state rlglue.State
	rng   *rand.Rand

	offlineDataObs [][]float64
	offlineStarts  []int
	offlineTermns  []int

	trueDataObs [][]float64
	trueStarts  []int
	trueTermns  []int

	offlineModel    *transModel.TransNetwork

	stateDim        int
	NumberOfActions int
	stateRange      []float64
	rewardBound     []float64
	stateBound      [][]float64
	PickStartFunc   EpStartFuncNet

	maxSDist  float64
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
	//traceLog := folder + "/traces-" + strconv.Itoa(int(run)) + ".csv"
	traceLog := folder + "/traces-" + strconv.Itoa(int(run%env.networkSettings.TotalLogs)) + ".csv"
	trueStartLog := trueStartFolder + "/traces-" + strconv.Itoa(int(run%env.networkSettings.TotalLogs)) + ".csv"
	//fmt.Println(traceLog)
	//fmt.Println(trueStartLog)

	env.Message("network data log", traceLog, "\n")
	env.Message("network starts log", trueStartLog, "\n")
	paramLog := folder + "/log_json.txt"
	env.SettingFromLog(paramLog)
	env.state = make(rlglue.State, env.stateDim)

	//env.offlineData = allTrans
	env.offlineDataObs = env.LoadData(traceLog)
	fmt.Println("Offline Data Loaded")
	//fmt.Println("Offline Dataset: ", env.offlineDataObs)
	env.trueDataObs = env.LoadData(trueStartLog)
	fmt.Println("True Data Loaded")

	env.trueStarts, env.trueTermns = env.SearchOfflineStart(env.trueDataObs)

	env.offlineModel = transModel.New()
	env.offlineModel.Initialize(env.networkSettings.Seed, env.offlineDataObs, env.networkSettings.TrainEpoch, env.networkSettings.TrainBatchSize,
		env.networkSettings.TrainLearningRate, env.networkSettings.HiddenLayer, env.stateDim, env.NumberOfActions)
	env.offlineModel.Train()

	if env.networkSettings.PickStartS == "random-init" { // default setting
		env.PickStartFunc = env.randomizeInitState
	} else if env.networkSettings.PickStartS == "random-all" {
		env.PickStartFunc = env.randomizeState
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

	env.rewardBound = make([]float64, 2)
	env.rewardBound[0], _ = ao.ArrayMin(rewards)
	env.rewardBound[1], _ = ao.ArrayMax(rewards)
	fmt.Println("Rewards min, max:", env.rewardBound[0], env.rewardBound[1])
	env.stateBound = make([][]float64, 2)
	for i := 0; i < len(allStates[0]); i++ {
		mn, _ := ao.ColumnMin(allStates, i)
		mx, _ := ao.ColumnMax(allStates, i)
		env.stateBound[0] = append(env.stateBound[0], mn)
		env.stateBound[1] = append(env.stateBound[1], mx)
	}
	fmt.Println("States min, max:", env.stateBound[0], env.stateBound[1])
	env.maxSDist = env.Distance(env.stateBound[0], env.stateBound[1])

	if env.NoisyS != 0 {
		for i := 0; i < len(allTransObs); i++ {
			temp := env.AddStateNoise(allTransObs[i][:env.stateDim], env.stateBound)
			copy(allTransObs[i][:env.stateDim], temp)
			temp = env.AddStateNoise(allTransObs[i][env.stateDim+1:env.stateDim*2+1], env.stateBound)
			copy(allTransObs[i][env.stateDim+1:env.stateDim*2+1], temp)
		}
	}

	var allTransObsKeep [][]float64
	//if env.EnsembleSeed != 0 {
	//	tempRnd := rand.New(rand.NewSource(int64(env.EnsembleSeed)))
	if env.DropPerc != 0 {
		//fmt.Println("HERE", env.DropPerc)
		filteredLen := int(float64(len(allTransObs)) * (1 - env.DropPerc))
		filteredIdx := env.rng.Perm(len(allTransObs))[:filteredLen]
		//fmt.Println("HERE", filteredIdx[:5])

		allTransObsKeep = make([][]float64, filteredLen)
		for i := 0; i < filteredLen; i++ {
			allTransObsKeep[i] = allTransObs[filteredIdx[i]]
		}
	} else {
		allTransObsKeep = allTransObs
	}
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

///* For debugging only */
//func (env *networkModelEnv) cheatingChoice() rlglue.State {
//	randIdx := env.rng.Intn(len(env.trueStarts))
//	state := env.offlineData[env.trueStarts[randIdx]-1]
//	fmt.Println("terminal ", state)
//	state = state[:env.stateDim]
//	return state
//}

func (env *networkModelEnv) randomizeState() rlglue.State {
	randIdx := env.rng.Intn(len(env.offlineDataObs))
	state := env.offlineDataObs[randIdx][:env.stateDim]
	return state
}

// Start returns an initial observation.
func (env *networkModelEnv) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	env.state = env.PickStartFunc()
	fmt.Println("Picked Start State: ", env.state)
	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)

	key := ao.FloatAryToString(env.state, " ")
	fmt.Println("Key: ", key)
	return state_copy, ""
}

func (env *networkModelEnv) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	actInt, _ := tpo.GetInt(act)
	nextState, reward, done := env.offlineModel.PredictSingleTrans(env.state, float64(actInt))
	env.state = nextState
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

func (env *networkModelEnv) Distance(state1 rlglue.State, state2 rlglue.State) float64 {
	var distance float64
	var squareddistance float64
	squareddistance = 0.0
	for i := 0; i < len(state1); i++ {
		squareddistance += math.Pow(state1[i]-state2[i], 2)
	}
	distance = math.Pow(squareddistance, 0.5)
	return distance
}

func (env *networkModelEnv) AddStateNoise(data []float64, bound [][]float64) []float64 {
	for j := 0; j < len(data); j++ {
		data[j] += env.rng.Float64() * env.NoisyS * (bound[1][j] - bound[0][j])
	}
	return data
}

func (env *networkModelEnv) GetInfo(info string, value float64) interface{} {
	return nil
}
