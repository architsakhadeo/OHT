import tensorflow.compat.v1 as tf
import os
import sys
tf.disable_v2_behavior() 

def graph_construction(save_name, alpha="0.0001", seed="0"):
    tf.set_random_seed(int(seed))
    alpha = float(alpha)
    # with tf.variable_scope("behaviour"):
    # Behaviour
    b_x = tf.placeholder(tf.float32, shape=[None, 4], name='beh_in')
    # b_y = tf.placeholder(tf.float32, shape=[None, 2], name='beh_truth')

    b_ly1= tf.layers.dense(b_x, 128, tf.nn.relu, name='beh_ly1')
    b_ly2 = tf.layers.dense(b_ly1, 128, tf.nn.relu, name='beh_ly2')
    b_ly3 = tf.layers.dense(b_ly2, 2, name='beh_ly3')
    b_y_ = tf.identity(b_ly3, name='beh_out')

    b_action = tf.placeholder(tf.float32, shape=[None, 1], name='beh_action_in')
    a_indices = tf.stack([tf.range(tf.shape(b_action)[0], dtype=tf.int32), 
                          tf.reshape(tf.cast(b_action, tf.int32), [tf.shape(b_action)[0]])], 
                          axis=1)
    b_y_act = tf.gather_nd(params=b_y_, indices=a_indices, name='beh_out_act')
    b_y_argmax = tf.math.argmax(b_y_, axis=1, name='beh_out_argmax')
 
    # with tf.variable_scope("target"):
    # Target
    t_x = tf.placeholder(tf.float32, shape=[None, 4], name='target_in')
    t_x_nog = tf.stop_gradient(t_x, name="target_in_no_g")
    t_y = tf.placeholder(tf.float32, shape=[None, 2], name='target_truth')

    t_ly1 = tf.layers.dense(t_x_nog, 128, tf.nn.relu, name='target_ly1')
    t_ly2 = tf.layers.dense(t_ly1, 128, tf.nn.relu)#, name='target_ly2')
    t_ly3 = tf.layers.dense(t_ly2, 2)#, name='target_ly_3')
    t_y_ = tf.identity(t_ly3, name='target_out')
    t_y_max = tf.math.reduce_max(t_y_, axis=1, keepdims=True, name='target_out_max')

    reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')
    gamma = tf.placeholder(tf.float32, shape=[None, 1], name='gamma')
    target = tf.math.add(reward, tf.math.multiply(gamma, t_y_max), name='target')

    # Optimize loss
    # b_loss = tf.reduce_mean(tf.square(b_y_ - b_y), name='beh_loss')
    b_loss = tf.reduce_mean(tf.square(b_y_act - target), name='beh_loss')
    # b_optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    b_optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
    b_train_op = b_optimizer.minimize(b_loss, name='beh_train')

    # with tf.variable_scope("sync"):
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(b_ly1.name)[0] + '/kernel:0')
    set1 = tf.assign(tf.get_default_graph().get_tensor_by_name(os.path.split(t_ly1.name)[0] + '/kernel:0'), weights, name="set1")
    
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(b_ly2.name)[0] + '/kernel:0')
    set2 = tf.assign(tf.get_default_graph().get_tensor_by_name(os.path.split(t_ly2.name)[0] + '/kernel:0'), weights, name="set2")

    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(b_ly3.name)[0] + '/kernel:0')
    set3 = tf.assign(tf.get_default_graph().get_tensor_by_name(os.path.split(t_ly3.name)[0] + '/kernel:0'), weights, name="set3")

    


    init = tf.global_variables_initializer()

    saver_def = tf.train.Saver().as_saver_def()

    # print('Run this operation to initialize variables     : ', init.name)
    # print('Run this operation for a train step            : ', b_train_op.name)

    # Write the graph out to a file.
    with open(save_name, 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())
        print("Graph saved")


    # print('''
    #     // save saves the current value of variables in graph/sess in files with the
    #     // given prefix and returns the string to provide to restore.
    #     func save(graph *tf.Graph, sess *tf.Session, prefix string) (string, error) {
    #     t, err := tf.NewTensor(prefix)
    #     if err != nil {
    #         return "", err
    #     }
    #     o := graph.Operation("%s").Output(0)
    #     ret, err := sess.Run(map[tf.Output]*tf.Tensor{o:t}, []tf.Output{graph.Operation("%s").Output(0)}, nil)
    #     if err != nil {
    #         return "", err
    #     }
    #     return ret[0].Value().(string), nil
    #     }
    #     // restore restores the value of variables previously saved using save.
    #     func restore(graph *tf.Graph, sess *tf.Session, path string) error {
    #     t, err := tf.NewTensor(path)
    #     if err != nil {
    #         return err
    #     }
    #     o := graph.Operation("%s").Output(0)
    #     _, err = sess.Run(map[tf.Output]*tf.Tensor{o:t}, nil, []*tf.Operation{graph.Operation("%s")})
    #     return err
    #     }
    #     ''') % (sd.filename_tensor_name[:-2], sd.save_tensor_name[:-2], sd.filename_tensor_name[:-2], sd.restore_op_name)

#     a = np.array([[1,2,3,4], [5,5,6,7]]).astype(np.float32)
#     # at = tf.convert_to_tensor(a0)
#     b = [[0.1, 0.2], [0.3, 0.4]]
#     with tf.Session() as sess:
#         sess.run(init)
#         sess.run(b_y_, feed_dict={b_x: a})
#         print(b_y_.eval())



# import numpy as np
# path = "data/nn/graph.pb"
# graph_construction(path)
# #with open(save_name, 'wb') as f:
# with tf.Session() as sess:
#     tf.saved_model.load(sess, None, path)