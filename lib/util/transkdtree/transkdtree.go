package transModel

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/kdtree"
)


/* helper */
type nodes []node
func (n nodes) Index(i int) kdtree.Comparable         { return n[i] }
func (n nodes) Len() int                              { return len(n) }
func (n nodes) Pivot(d kdtree.Dim) int                { return plane{nodes: n, Dim: d}.Pivot() }
func (n nodes) Slice(start, end int) kdtree.Interface { return n[start:end] }


/* helper */
type node struct {
	key []float64
	value []float64
}
func (n node) Compare(c kdtree.Comparable, d kdtree.Dim) float64 {
	q := c.(node)
	vectorX := mat.NewDense(len(q.key), 1, q.key)
	vectorY := mat.NewDense(len(n.key), 1, n.key)
	var temp mat.Dense
	temp.Sub(vectorX, vectorY)
	dist := mat.Norm(&temp, 2)
	return dist
}
func (n node) Dims() int { return len(n.key) }
func (n node) Distance(c kdtree.Comparable) float64 {
	q := c.(node)
	vectorX := mat.NewDense(len(q.key), 1, q.key)
	vectorY := mat.NewDense(len(n.key), 1, n.key)
	var temp mat.Dense
	temp.Sub(vectorX, vectorY)
	dist := mat.Norm(&temp, 2)
	return dist
}


/* helper */
type plane struct {
	kdtree.Dim
	nodes
}
func (p plane) Less(i, j int) bool {
	return p.nodes[i].key[p.Dim] < p.nodes[j].key[p.Dim]
}
func (p plane) Pivot() int { return kdtree.Partition(p, kdtree.MedianOfMedians(p)) }
func (p plane) Slice(start, end int) kdtree.SortSlicer {
	p.nodes = p.nodes[start:end]
	return p
}
func (p plane) Swap(i, j int) {
	p.nodes[i], p.nodes[j] = p.nodes[j], p.nodes[i]
}


/* KD-tree */
type TransTrees struct {
	trees        []*kdtree.Tree
	data         [][][]float64
	count        []int
	numAction    int
	stateDim     int
	//distanceFunc pairwise.PairwiseDistanceFunc
}

func New(numAction int, actionIdx int) TransTrees {
	ts := make([]*kdtree.Tree, numAction)
	data := make([][][]float64, numAction)
	cs := make([]int, numAction)
	t := TransTrees{ts, data, cs, numAction, actionIdx}
	return t
}

func (t *TransTrees) BuildTree(allTrans [][]float64) {
	//for i := 0; i < numAction; i++ {
	//	ts[i] = kdtree.New()
	//	cs[i] = 0
	//}
	var action int
	dataInTree := make([][]node, t.numAction)
	for i := 0; i < len(allTrans); i++ {
		action = int(allTrans[i][t.stateDim])
		t.data[action] = append(t.data[action], allTrans[i])                      // sort current state
		dataInTree[action] = append(dataInTree[action], node{allTrans[i][:t.stateDim], allTrans[i]}) // sort current state
	}
	for i := 0; i < t.numAction; i++ {
		t.trees[i] = kdtree.New(nodes(dataInTree[i]), false)
		t.count[i] = len(dataInTree[i])
	}
}

func (t *TransTrees) SearchTree(target []float64, action int, k int) ([][]float64, [][]float64, []float64, []float64, []float64) {
	var keep kdtree.Keeper
	q := node{target, nil} // we don't need value for a compared node

	/* all neighbors in a fixed range */
	//keep = kdtree.NewDistKeeper(10000.0) // Give a large enough upper bound
	//t.trees[action].NearestSet(keep, q)
	//for _, c := range keep.(*kdtree.DistKeeper).Heap {
	//	p := c.Comparable.(node)
	//	fmt.Println(p.data, p.Distance(q))
	//}
	//fmt.Println()

	// Find the k closest transitions to the target.
	keep = kdtree.NewNKeeper(k)
	t.trees[action].NearestSet(keep, q)

	states := make([][]float64, k)
	nextStates := make([][]float64, k)
	rewards := make([]float64, k)
	terminals := make([]float64, k)
	dists := make([]float64, k)
	//fmt.Println(`k closest transitions to`, target)
	for i, c := range keep.(*kdtree.NKeeper).Heap {
		p := c.Comparable.(node)
		states[i] = p.value[:t.stateDim]
		nextStates[i] = p.value[t.stateDim+1 : t.stateDim*2+1]
		rewards[i] = p.value[t.stateDim*2+1]
		terminals[i] = p.value[t.stateDim*2+2]
		dists[i] = p.Distance(q)

		//fmt.Println(i, p.key, p.Distance(q))
	}
	//fmt.Println()

	return states, nextStates, rewards, terminals, dists
}

//func test() {
//
//	raw := [][]float64{{2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}}
//	dist := pairwise.NewEuclidean()
//	t := transkdtree.New()
//	t.Build(raw)
//	q := []float64{8, 7}
//	p, d, _ := t.Search(1, dist, q)
//	fmt.Printf("%v is closest point to %v, d=%f\n", p, q, d)
//}
