package buffer

import (
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

// type BufferControl interface {
// 	Initialize(size int, statelen int)
// 	Feed(laststate rlglue.State, lastaction rlglue.Action, state rlglue.State, reward int, gamma float64)
// 	Sample(batchsize int) [][]float64
// }

type Buffer struct {
	sample_type string
	size        int
	state_len   int
	seq_len     int
	idx         int
	queue       [][]float64
	rng         *rand.Rand
}

// func init() {
// 	// fmt.Println("Init Buffer")
// }

func NewBuffer() *Buffer {
	return &Buffer{}
}

func (bf *Buffer) Initialize(stype string, size int, slen int, seed int64) {
	bf.sample_type = stype
	bf.size = size
	bf.queue = make([][]float64, size)
	bf.state_len = slen
	bf.seq_len = slen*2 + 3
	for i := range bf.queue {
		bf.queue[i] = make([]float64, bf.seq_len)
	}
	bf.rng = rand.New(rand.NewSource(seed))
}

func (bf *Buffer) Feed(laststate rlglue.State, lastaction int, state rlglue.State, reward float64, gamma float64) {
	seq := make([]float64, bf.seq_len)
	copy(seq[:bf.state_len], laststate)
	seq[bf.state_len] = float64(lastaction)
	copy(seq[bf.state_len+1:bf.state_len*2+1], state)
	seq[bf.state_len*2+1] = reward
	seq[bf.state_len*2+2] = gamma

	bf.queue[bf.idx%bf.size] = seq
	bf.idx += 1
}

func (bf *Buffer) Array2Trans(samples [][]float64) ([][]float64, []int, [][]float64, []float64, []float64){
	lastStates := ao.Index2d(samples, 0, len(samples), 0, bf.state_len)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(ao.Index2d(samples, 0, len(samples), bf.state_len, bf.state_len+1)))
	states := ao.Index2d(samples, 0, len(samples), bf.state_len+1, bf.state_len*2+1)
	rewards := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), bf.state_len*2+1, bf.state_len*2+2))
	gammas := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), bf.state_len*2+2, bf.state_len*2+3))
	return lastStates, lastActions, states, rewards, gammas
}

func (bf *Buffer) Sample(batchsize int) ([][]float64, []int, [][]float64, []float64, []float64){
	var samples [][]float64
	if bf.sample_type == "random" {
		samples = bf.RandomSample(batchsize)
	}
	return bf.Array2Trans(samples)
}

func (bf *Buffer) RandomSample(batchsize int) [][]float64 {
	samples := make([][]float64, batchsize)
	for i := range samples {
		samples[i] = make([]float64, bf.state_len)
	}

	var sample_range int = bf.idx
	if bf.idx > bf.size {
		sample_range = bf.size
	}
	for i := 0; i < batchsize; i++ {
		samples[i] = bf.queue[bf.rng.Intn(sample_range)]
	}

	return samples
}

func (bf *Buffer) Content() ([][]float64, []int, [][]float64, []float64, []float64){
	if bf.idx < bf.size {
		return bf.Array2Trans(bf.queue[:bf.idx])
	} else {
		return bf.Array2Trans(bf.queue)
	}
}

func (bf *Buffer) GetLength() int {
	if bf.idx < bf.size {
		return bf.idx
	} else {
		return bf.size
	}
}