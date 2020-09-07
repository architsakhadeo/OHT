package environment

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	transModel "github.com/stellentus/cartpoles/lib/util/kdtree"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

const (

)

type knnSettings struct {
	Seed         int64     `json:"seed"`
	offlineLog   string    `json:"trans-log"`
	NumberOfActions     int     `json:"numberOfActions"`
	stateRange   []float64 `json:"stateRange"`
}

type KnnModelEnv struct {
	logger.Debug
	knnSettings
	state             rlglue.State
	rng               *rand.Rand
	offlineData		  [][]float64
	offlineModel	  transModel.TransTrees
	stateDim		  int
}

func init() {
	Add("knnModel", NewKnnModelEnv)
}

func NewKnnModelEnv(logger logger.Debug) (rlglue.Environment, error) {
	return &KnnModelEnv{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *KnnModelEnv) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.knnSettings)
	if err != nil {
		err = errors.New("environment.Cartpole settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	env.Seed += int64(run)
	env.rng = rand.New(rand.NewSource(env.Seed)) // Create a new rand source for reproducibility
	env.state = make(rlglue.State, 4)

	env.Message("offline knn model environment settings", fmt.Sprintf("%+v", env.knnSettings))

	folder := env.knnSettings.offlineLog
	traceLog := folder+"/trace-"+string(run)+".csv"
	paramLog := folder + "/log_json.txt"

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
		fmt.Println()
		line = scanner.Text()
		spl = strings.Split(line, "=")
		if spl[0] == "state-len" { //stateDimension
			stateDim, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "numberOfActions" {
			numAction, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "stateRange" {
			strRange := strings.Split(spl[1][1:len(spl[1])-1], ", ")
			stateRange = make([]float64, len(strRange))
			for i:=0; i<len(strRange); i++ {
				stateRange[i], _ = strconv.ParseFloat(strRange[i], 64)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	env.stateDim = stateDim
	env.NumberOfActions = numAction
	env.stateRange = stateRange

	// Get offline data
	csvFile, err := os.Open(traceLog)
	if err != nil {
		log.Fatal(err)
	}
	allTransStr, err := csv.NewReader(csvFile).ReadAll()
	csvFile.Close()
	if err != nil {
		log.Fatal(err)
	}
	allTrans := make([][]float64, len(allTransStr))
	for i:=0; i<len(allTransStr); i++ {
		trans := allTransStr[i]
		row := make([]float64, len(trans))
		for j, num := range trans {
			row[j], _ = strconv.ParseFloat(num, 64)
		}
		allTrans[i] = row
	}
	env.offlineData = allTrans
	env.offlineModel = transModel.New(numAction, stateDim, "euclidean")
	env.offlineModel.BuildTree(allTrans)
	return nil
}

func (env *KnnModelEnv) randomizeState() rlglue.State {
	randIdx := env.rng.Int()
	return env.offlineData[randIdx][:env.stateDim]
}

// Start returns an initial observation.
func (env *KnnModelEnv) Start() rlglue.State {
	env.state = env.randomizeState()
	cp := make([]float64, env.stateDim)
	copy(cp, env.state)
	return cp
}

func (env *KnnModelEnv) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	_, nextStates, rewards, gammas, _ := env.offlineModel.SearchTree(env.state, int(act), 1)
	cp := make([]float64, env.stateDim)
	env.state = nextStates[0]
	copy(cp, env.state)
	var done bool
	if gammas[0] == 0 {
		done = true
	} else {
		done = false
	}
	return cp, rewards[0], done
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
