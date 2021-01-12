package transModel

import (
	"github.com/sjwhitworth/golearn/kdtree"
	"github.com/sjwhitworth/golearn/metrics/pairwise"
)

type TransTrees struct {
	trees        []*kdtree.Tree
	data         [][][]float64
	count        []int
	numAction    int
	stateDim     int
	distanceFunc pairwise.PairwiseDistanceFunc
}

func New(numAction int, actionIdx int, dist string) TransTrees {
	ts := make([]*kdtree.Tree, numAction)
	data := make([][][]float64, numAction)
	cs := make([]int, numAction)
	for i := 0; i < numAction; i++ {
		ts[i] = kdtree.New()
		cs[i] = 0
	}
	var dfunc pairwise.PairwiseDistanceFunc
	if dist == "euclidean" {
		dfunc = pairwise.NewEuclidean()
	} else {
		panic("Undefined distance function")
	}
	t := TransTrees{ts, data, cs, numAction, actionIdx, dfunc}
	return t
}

func (t *TransTrees) BuildTree(allTrans [][]float64) {
	var action int
	dataInTree := make([][][]float64, t.numAction)
	for i := 0; i < len(allTrans); i++ {
		action = int(allTrans[i][t.stateDim])
		t.data[action] = append(t.data[action], allTrans[i])                      // sort current state
		dataInTree[action] = append(dataInTree[action], allTrans[i][:t.stateDim]) // sort current state
	}
	for i := 0; i < t.numAction; i++ {
		err := t.trees[i].Build(dataInTree[i])
		//fmt.Println("Data in Build Tree: ", len(dataInTree[i]))
		if err != nil {
			panic("Error when building tree: " + string(err.Error()))
		}
	}
}

func (t *TransTrees) SearchTree(target []float64, action int, k int) ([][]float64, [][]float64, []float64, []float64, []float64) {
	neighborIdxs, dists, err := t.trees[action].Search(k, t.distanceFunc, target) // search by current state
	if err != nil {
		panic(err)
	}
	states := make([][]float64, k)
	nextStates := make([][]float64, k)
	rewards := make([]float64, k)
	terminals := make([]float64, k)
	//fmt.Println("Indices: ", neighborIdxs)
	for i := 0; i < k; i++ {
		states[i] = t.data[action][neighborIdxs[i]][:t.stateDim]
		nextStates[i] = t.data[action][neighborIdxs[i]][t.stateDim+1 : t.stateDim*2+1]
		rewards[i] = t.data[action][neighborIdxs[i]][t.stateDim*2+1]
		terminals[i] = t.data[action][neighborIdxs[i]][t.stateDim*2+2]
	}
	return states, nextStates, rewards, terminals, dists
}

//func test() {
//
//	raw := [][]float64{{2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}}
//	dist := pairwise.NewEuclidean()
//	t := kdtree.New()
//	t.Build(raw)
//	q := []float64{8, 7}
//	p, d, _ := t.Search(1, dist, q)
//	fmt.Printf("%v is closest point to %v, d=%f\n", p, q, d)
//}
