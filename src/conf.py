from dotmap import DotMap as Map
import tensorflow as tf


INP_SIZE, OUT_SIZE = 10, 2

network_1 = Map()
network_1.conv1 = Map()
network_1.conv1.filter_shape = [5, 5]
network_1.conv1.strides = [5, 5]
network_1.conv2.filter_shape = [5, 5]
network_1.conv2.strides = [5, 5]
network_1.fc1.out_units = 100
network_1.fc1.out_units = 5


mlp = Map()
mlp.fc1 = Map()
mlp.fc1.weights = tf.truncated_normal(shape=(10 * 56, 400), stddev=0.01) # 56, 75, 111, 147
# [147, 133, 118, 103, 89, 74, 59, 45, 30, 15]
mlp.fc1.activation = tf.nn.relu
mlp.fc2 = Map()
mlp.fc2.weights = tf.truncated_normal(shape=(400, 400), stddev=0.01)
mlp.fc2.activation = tf.nn.relu
mlp.fc3 = Map()
mlp.fc3.weights = tf.truncated_normal(shape=(400, 30 * 80), stddev=0.01)
mlp.fc3.activation = tf.identity
mlp.reshape = Map()
mlp.reshape.shape = [None, 80, 30, 1]
mlp.loss_type = "sigmoid_cross_entropy"
mlp.optimizer = tf.train.AdamOptimizer(0.0005, epsilon=1e-14)









mlp_discrete = Map()
mlp_discrete.fc1 = Map()
mlp_discrete.fc1.weights = tf.truncated_normal(shape=(11*147,100), stddev=0.01) # 36, 56, 75, 111, 147
mlp_discrete.fc1.activation = tf.nn.relu
mlp_discrete.fc2 = Map()
mlp_discrete.fc2.weights = tf.truncated_normal(shape=(100, 100), stddev=0.01)
mlp_discrete.fc2.activation = tf.nn.relu
mlp_discrete.fc3 = Map()
mlp_discrete.fc3.weights = tf.truncated_normal(shape=(100, 5), stddev=0.01)
mlp_discrete.fc3.activation = tf.identity
mlp_discrete.reshape = Map()
mlp_discrete.reshape.shape = [None, 5]
mlp_discrete.loss_type = "sigmoid_cross_entropy"
mlp_discrete.optimizer = tf.train.AdamOptimizer(0.0005, epsilon=1e-14)
# mlp_discrete.optimizer = tf.train.GradientDescentOptimizer(0.005)

