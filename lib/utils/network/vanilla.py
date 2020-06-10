import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior() 

def graph_construction(save_name):
    # with tf.variable_scope("behaviour"):
    # Behaviour
    b_x = tf.placeholder(tf.float32, shape=[None, 4], name='beh_in')
    # b_y = tf.placeholder(tf.float32, shape=[None, 2], name='beh_truth')

    b_ly1= tf.layers.dense(b_x, 32, tf.nn.relu, name='beh_ly1')
    b_ly2 = tf.layers.dense(b_ly1, 32, tf.nn.relu, name='beh_ly2')
    b_ly3 = tf.layers.dense(b_ly2, 2, name='beh_ly3')
    b_y_ = tf.identity(b_ly3, name='beh_out')

    b_action = tf.placeholder(tf.float32, shape=[None, 1], name='beh_action_in')
    a_indices = tf.stack([tf.range(tf.shape(b_action)[0], dtype=tf.int32), 
                          tf.reshape(tf.cast(b_action, tf.int32), [tf.shape(b_action)[0]])], 
                          axis=1)
    b_y_act = tf.gather_nd(params=b_y_, indices=a_indices, name='beh_out_act')

    # with tf.variable_scope("target"):
    # Target
    t_x = tf.placeholder(tf.float32, shape=[None, 4], name='target_in')
    t_x_nog = tf.stop_gradient(t_x, name="target_in_no_g")
    t_y = tf.placeholder(tf.float32, shape=[None, 2], name='target_truth')

    t_ly1 = tf.layers.dense(t_x_nog, 32, tf.nn.relu)#, name='target_ly1')
    t_ly2 = tf.layers.dense(t_ly1, 32, tf.nn.relu)#, name='target_ly2')
    t_ly3 = tf.layers.dense(t_ly2, 2)#, name='target_ly_3')
    t_y_ = tf.identity(t_ly3, name='target_out')
    t_y_max = tf.math.reduce_max(t_y_, axis=1, keepdims=True, name='target_out_max')

    reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')
    gamma = tf.placeholder(tf.float32, shape=[None, 1], name='gamma')
    target = tf.math.add(reward, tf.math.multiply(gamma, t_y_max), name='target')

    # Optimize loss
    # b_loss = tf.reduce_mean(tf.square(b_y_ - b_y), name='beh_loss')
    b_loss = tf.reduce_mean(tf.square(b_y_act - target), name='beh_loss')
    b_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    b_train_op = b_optimizer.minimize(b_loss, name='beh_train')

    # with tf.variable_scope("sync"):
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(b_ly1.name)[0] + '/kernel:0')
    set1 = tf.assign(tf.get_default_graph().get_tensor_by_name(os.path.split(t_ly1.name)[0] + '/kernel:0'), weights, name="set1")
    
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(b_ly2.name)[0] + '/kernel:0')
    set2 = tf.assign(tf.get_default_graph().get_tensor_by_name(os.path.split(t_ly2.name)[0] + '/kernel:0'), weights, name="set2")

    # weights = tf.get_default_graph().get_tensor_by_name(os.path.split(b_y_.name)[0] + '/kernel:0')
    # set3 = tf.assign(tf.get_default_graph().get_tensor_by_name(os.path.split(t_y_.name)[0] + '/kernel:0'), weights, name="set3")
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

    # a = np.array([[1,2,3,4], [5,5,6,7]]).astype(np.float32)
    # b = [[0.1, 0.2], [0.3, 0.4]]
    # at = tf.convert_to_tensor(a)
    # bt = tf.convert_to_tensor(b)
    # with tf.Session() as sess:
    #     sess.run(init)
    #     sess.run(t_y_, feed_dict={t_x: a})
    #     print(t_y_)



# import numpy as np
# path = "data/nn/graph.pb"
# graph_construction(path)
