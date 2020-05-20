package buffer

import (
	"math/rand"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

// type BufferControl interface {
// 	Initialize(size int, statelen int)
// 	Feed(laststate rlglue.State, lastaction rlglue.Action, state rlglue.State, reward int, gamma float64)
// 	Sample(batchsize int) [][]float64
// }

type Buffer struct {
	sample_type		string 
	size			int	
	state_len		int
	seq_len			int
	idx				int
	queue			[][]float64
}

// func init() {
// 	// fmt.Println("Init Buffer") 
// }

func NewBuffer() (*Buffer) {
	return &Buffer{}
}

func (bf *Buffer) Initialize(stype string, size int, slen int) {
	bf.sample_type = stype
	bf.size = size
	bf.queue = make([][]float64, size)
	bf.state_len = slen
	bf.seq_len = slen * 2 + 3
	for i := range bf.queue {
		bf.queue[i] = make([]float64, bf.seq_len)
	}
}

func (bf *Buffer) Feed(laststate rlglue.State, lastaction int, state rlglue.State, reward float64, gamma float64) {
// func (bf *Buffer) Feed(laststate []float64, lastaction int, state []float64, reward float64, gamma float64) {
	seq := make([]float64, bf.seq_len) 
	copy(seq[:bf.state_len], laststate)
	seq[bf.state_len] = float64(lastaction)
	copy(seq[bf.state_len+1: bf.state_len*2+1], state)
	seq[bf.state_len*2+1] = reward
	seq[bf.state_len*2+2] = gamma

	bf.queue[bf.idx % bf.size] = seq
	bf.idx += 1
}

func (bf *Buffer) Sample(batchsize int) [][]float64{
	var samples [][]float64
	if (bf.sample_type=="random") {
		samples = bf.RandomSample(batchsize)
	}
	return samples
}

func (bf *Buffer) RandomSample(batchsize int) [][]float64 {
	samples := make([][]float64, batchsize)
	for i := range samples {
		samples[i] = make([]float64, bf.state_len)
	}

	var sample_range int = bf.idx
	if (bf.idx > bf.size) {
		sample_range = bf.size
	}
	for i := 0; i < batchsize; i++ {
		samples[i] = bf.queue[rand.Intn(sample_range)]
	}

	return samples
}