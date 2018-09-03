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
mlp.fc1.weights = tf.truncated_normal(shape=(68*20,128), stddev=0.01) # 166 * 17
mlp.fc1.activation = tf.nn.relu
mlp.fc2 = Map()
mlp.fc2.weights = tf.truncated_normal(shape=(128, 128), stddev=0.01)
mlp.fc2.activation = tf.nn.relu
mlp.fc3 = Map()
mlp.fc3.weights = tf.truncated_normal(shape=(128, 80 * 30), stddev=0.01)
mlp.fc3.activation = tf.identity
mlp.reshape = Map()
mlp.reshape.shape = [None, 80, 30, 1]
mlp.loss_type = "sigmoid_cross_entropy"
mlp.optimizer = tf.train.AdamOptimizer(0.001)

cnn1 = Map()
cnn1.conv1 = Map()
cnn1.conv1.weights = tf.truncated_normal(shape=(166, 1, 1, 30), stddev=0.01)
cnn1.conv1.strides = [1, 1, 1, 1]
cnn1.conv1.padding = "VALID"
cnn1.conv1.activation = tf.nn.relu
cnn1.conv2 = Map()
cnn1.conv2.weights = tf.truncated_normal(shape=(1, 5, 30, 4), stddev=0.01)
cnn1.conv2.strides = [1, 1, 1, 1]
cnn1.conv2.padding = "VALID"
cnn1.conv2.activation = tf.nn.relu
cnn1.convT1 = Map()
cnn1.fc1.weights = tf.truncated_normal(shape=(16 * 4, 128), stddev=0.01)
cnn1.fc1.activation = tf.nn.relu
cnn1.fc2 = Map()
cnn1.fc2.weights = tf.truncated_normal(shape=(128, 48 * 18), stddev=0.01)
cnn1.fc2.activation = tf.identity
cnn1.reshape = Map()
cnn1.reshape.shape = [None, 48, 18, 1]
cnn1.loss_type = "sigmoid_cross_entropy"
cnn1.optimizer = tf.train.AdamOptimizer(0.001)

sp1 = Map()
sp1.conv1 = Map()
sp1.conv1.weights = tf.truncated_normal(shape=(68, 5, 1, 32), stddev=0.1)
sp1.conv1.strides = [1, 1, 1, 1]
sp1.conv1.padding = "VALID"
sp1.conv1.activation = tf.nn.relu
sp1.conv2 = Map()
sp1.conv2.weights = tf.truncated_normal(shape=(1, 5, 32, 128), stddev=0.1)
sp1.conv2.strides = [1, 1, 1, 1]
sp1.conv2.padding = "VALID"
sp1.conv2.activation = tf.nn.relu
sp1.convT1 = Map()
sp1.convT1.weights = tf.truncated_normal(shape=(39, 11, 32, 128), stddev=0.1)
sp1.convT1.output_shape = [None, 39, 20, 32]
sp1.convT1.strides = [1, 1, 1, 1]
sp1.convT1.padding = "VALID"
sp1.convT1.activation = tf.nn.relu
sp1.convT2 = Map()
sp1.convT2.weights = tf.truncated_normal(shape=(4, 4, 1, 32), stddev=0.1)
sp1.convT2.output_shape = [None, 80, 23, 1]
sp1.convT2.strides = [1, 2, 1, 1]
sp1.convT2.padding = "VALID"
sp1.convT2.activation = tf.identity
sp1.loss_type = "rmse"
sp1.optimizer = tf.train.GradientDescentOptimizer(0.01)


sp2 = Map()
sp2.conv1 = Map()
sp2.conv1.weights = tf.truncated_normal(shape=(68, 1, 1, 100), stddev=0.01)
sp2.conv1.strides = [1, 1, 1, 1]
sp2.conv1.padding = "VALID"
sp2.conv1.activation = tf.nn.relu
sp2.conv2 = Map()
sp2.conv2.weights = tf.truncated_normal(shape=(1, 5, 100, 150), stddev=0.01)
sp2.conv2.strides = [1, 1, 1, 1]
sp2.conv2.padding = "VALID"
sp2.conv2.activation = tf.nn.relu
sp2.convT1 = Map()
sp2.fc1.weights = tf.truncated_normal(shape=(14 * 150, 128), stddev=0.01)
sp2.fc1.activation = tf.nn.relu
sp2.fc2 = Map()
sp2.fc2.weights = tf.truncated_normal(shape=(128, 80 * 26), stddev=0.01)
sp2.fc2.activation = tf.identity
sp2.reshape = Map()
sp2.reshape.shape = [None, 80, 26, 1]
sp2.loss_type = "sigmoid_cross_entropy"
sp2.optimizer = tf.train.AdamOptimizer(0.01)
