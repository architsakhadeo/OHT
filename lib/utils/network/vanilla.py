import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior() 

def graph_construction(save_name):
    # with tf.variable_scope("behaviour"):
    # Behaviour
    b_x = tf.placeholder(tf.float32, shape=[None, 1, 6], name='beh_in')
    b_y = tf.placeholder(tf.float32, shape=[None, 1, 4], name='beh_truth')

    b_ly1= tf.layers.dense(b_x, 32, tf.nn.relu, name='beh_ly1')
    b_ly2 = tf.layers.dense(b_ly1, 32, tf.nn.relu, name='beh_ly2')
    b_ly3 = tf.layers.dense(b_ly2, 4, name='beh_ly3')
    b_y_ = tf.identity(b_ly3, name='beh_out')

    # Optimize loss
    b_loss = tf.reduce_mean(tf.square(b_y_ - b_y), name='beh_loss')
    b_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    b_train_op = b_optimizer.minimize(b_loss, name='beh_train')

    # with tf.variable_scope("target"):
    # Target
    t_x = tf.placeholder(tf.float32, shape=[None, 1, 6], name='target_in')
    t_x_nog = tf.stop_gradient(t_x, name="target_in_no_g")
    t_y = tf.placeholder(tf.float32, shape=[None, 1, 4], name='target_truth')

    t_ly1 = tf.layers.dense(t_x_nog, 32, tf.nn.relu)#, name='target_ly1')
    t_ly2 = tf.layers.dense(t_ly1, 32, tf.nn.relu)#, name='target_ly2')
    t_ly3 = tf.layers.dense(t_ly2, 4)#, name='target_ly_3')
    t_y_ = tf.identity(t_ly3, name='target_out')

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



# import tensorflow as tf 


# def graph_construction(in_shape, out_shape, hidden, num_ly, lr, save_name):
#     init = tf.keras.initializers.he_normal()
#     # behaviour_weights = {}
#     # target_weights = {}
    
#     with tf.compat.v1.variable_scope("behaviour"):
#         behaviour = tf.keras.Sequential()
#         for _ in range(num_ly):
#             behaviour.add(tf.keras.layers.Dense(hidden, kernel_initializer=init))
#             behaviour.add(tf.keras.layers.ReLU())
#         behaviour.add(tf.keras.layers.Dense(out_shape, kernel_initializer=init))
#         behaviour.build((None, in_shape))

#         opt_fn = tf.keras.optimizers.Adam(learning_rate=lr)
#         loss_fn = tf.keras.losses.MeanSquaredError()
#         behaviour.compile(optimizer=opt_fn,
#                     loss=loss_fn,
#                     metrics=['mse'])

#     with tf.compat.v1.variable_scope("target"):
#         target = tf.keras.Sequential()
#         for _ in range(num_ly):
#             target.add(tf.keras.layers.Dense(hidden, kernel_initializer=init))
#             target.add(tf.keras.layers.ReLU())
#         target.add(tf.keras.layers.Dense(out_shape, kernel_initializer=init))
#         target.build((None, in_shape))

#         # opt_fn = tf.keras.optimizers.Adam(learning_rate=lr)
#         # loss_fn = tf.keras.losses.MeanSquaredError()
#         # model.compile(optimizer=opt_fn,
#         #             loss=loss_fn,
#         #             metrics=['mse'])
#     with tf.compat.v1.variable_scope("sync"):
#         # copy_from = {}
#         # copy_to = {}
#         # for name in behaviour_weights.keys():
#         #     copy_from[name] = tf.placeholder('float32', target_weights[name].get_shape().as_list(), name=name)
#         #     copy_to[name] = target_weights[name].assign(copy_from[name])
        
#         sync = target.set_weights(behaviour.get_weights())


#     with open(save_name, 'wb') as f:
#         f.write(tf.compat.v1.get_default_graph().as_graph_def().SerializeToString())
#     print("Graph saved")


# graph_construction("data/nn/graph.pb")
