package neuralnet

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Vanilla struct {
	input_dim		int64
	output_dim		int64
	hidden			[]int64	
	activation		string
	graph			*tf.Graph
}

func NewVanilla() (*Vanilla) {
	return &Vanilla{}
}

func (vn *Vanilla) Initialize(input_dim int, output_dim int, hidden []int, activation string) {
	vn.input_dim = int64(input_dim)
	vn.output_dim = int64(output_dim)
	for i := range hidden {
		// assuming little endian
		vn.hidden[i] = int64(hidden[i])
	}
	vn.activation = activation
	vn.graph = vn.Construction()
}

func (vn *Vanilla) Construction() *tf.Graph{
	root := op.NewScope()
	// in := op.Placeholder(root, tf.Float, op.PlaceholderShape(tf.MakeShape(vn.input_dim)))
	// target := op.Placeholder(root, tf.Float, op.PlaceholderShape(tf.MakeShape(vn.output_dim)))
	// out := op.MatMul(root, in, in)

	// loss := op.Reduce_sum(op.Square(out - target))
	// train = op.Train.GradientDescentOptimizer(learning_rate).minimize(agent.loss)	

	// net := op.Sequential()
	// in_d := vn.input_dim
	// for i := 0; i < len(vn.hidden); i++ {
	// 	// net.add(tf.Keras.Layers.Dense(vn.hidden[i], input_dim=in_d, activation=vn.activation))
	// 	net.add(tf.Keras.Layers.Dense(vn.hidden[i]))
	// 	net.add(tf.Keras.Layers.Activation(vn.activation))
	// 	in_d = vn.hidden[i]
	// }
	// tf.Keras.Layers.Dense(vn.output_dim)

	graph, _ := root.Finalize()
	return graph
}

// func (vn *Vanilla) MSE(pred tf.Tensor, target tf.Tensor) {
// 	loss := tf.reduce_sum(tf.square(pred - target))
// }