import tensorflow as tf
import numpy as np


DTYPE = tf.float32



def conf_to_loss(loss_type, logits, labels):
    if loss_type == "softmax_cross_entropy":
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    elif loss_type == "sigmoid_cross_entropy":
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    elif loss_type == "mse":
        axis = list(range(1, len(logits.shape.as_list())))
        return tf.reduce_mean((logits - labels) ** 2, axis=axis)
    elif loss_type == "rmse":
        axis = list(range(1, len(logits.shape.as_list())))
        return tf.reduce_mean(tf.abs(logits - labels), axis=axis)
    else:
        raise ValueError("Unrecognized loss type: " + loss_type)


class Layers(dict):
    def __setitem__(self, key, value):
        print("{}{}:\tshape = {}".format(key, " " * (20 - len(key)), value.shape.as_list()))
        return dict.__setitem__(self, key, value)


class Network:
    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    def train(self, sess, x, y,dropout=1):
        return sess.run(self.train_op, feed_dict={self.input: x, self.output_target: y, self.dropout:dropout})

    def valid(self, sess, x,dropout=1.0):
        return sess.run(self.output, feed_dict={self.input: x,self.dropout:dropout})

    def get_weights(self, sess):
        return sess.run(self.weights)

    def get_loss(self, sess, x, y):
        return sess.run(self.loss, feed_dict={self.input: x, self.output_target: y})


class OneLayerPerceptron(Network):
    # conf.fc1.weights : initial weights of the net
    # conf.loss_type   : type of the loss
    # conf.optimizer   : a tensorflow optimizer object
    def __init__(self, input_shape, conf):
        self.weights = []
        self.input = tf.placeholder(shape=input_shape, dtype=DTYPE)
        input_flat = tf.reshape(self.input, (tf.shape(self.input)[0], -1))
        self.weights.append(tf.Variable(conf.fc1.weights, dtype=DTYPE))
        self.output = tf.matmul(input_flat, self.weights[-1])
        self.layers = [input_flat, self.output]
        self.output_target = tf.placeholder_like(self.output, dtype=DTYPE)
        self.loss = conf_to_loss(conf.loss_type, self.output, self.output_target)
        optimizer = conf.optimizer
        with tf.control_dependencies([optimizer.minimize(self.loss)]):
            self.train_op = tf.identity(self.loss)


class MultiLayerPerceptron(Network):
    # conf.fc1.weights    : initial weights of the layer
    # conf.fc1.activation : activation function of the layer
    # conf.fc2.weights    : initial weights of the layer
    # conf.fc2.activation : activation function of the layer
    # conf.fc?.weights    : initial weights of the layer
    # conf.fc?.activation : activation function of the layer
    # conf.loss_type      : type of the loss
    # conf.optimizer      : a tensorflow optimizer object
    def __init__(self, input_shape, conf):
        self.weights = []
        self.input = tf.placeholder(shape=input_shape, dtype=DTYPE)
        self.dropout = tf.placeholder_with_default(1.0, shape=())

        batch_size = tf.shape(self.input)[0]
        input_flat = tf.reshape(self.input, (batch_size, -1))
        self.layers = [input_flat]
        n_layers = len([k for k in conf if k.find("fc") == 0])
        for layer_number in range(1, n_layers + 1):
            layer = conf["fc{}".format(layer_number)]
            self.weights.append(tf.Variable(layer.weights, dtype=DTYPE))
            self.layers.append(layer.activation(tf.matmul(self.layers[-1], self.weights[-1])))
        self.layers.append(tf.nn.dropout(x=self.layers[-1], keep_prob=self.dropout))  # TODO

        shape = conf.reshape.shape
        shape[0] = batch_size
        self.layers.append(tf.reshape(self.layers[-1], shape))
        self.output = self.layers[-1]
        self.output_target = tf.placeholder(shape=self.output.shape, dtype=DTYPE)
        self.loss = conf_to_loss(conf.loss_type, self.output, self.output_target)
        optimizer = conf.optimizer
        with tf.control_dependencies([optimizer.minimize(self.loss)]):
            self.train_op = tf.identity(self.loss)

class ConvolutionalNeuralNetwork1(Network):
    def __init__(self, input_shape, conf):
        self.weights = {}
        self.layers = Layers()
        self.input = tf.placeholder(shape=input_shape, dtype=DTYPE)
        self.dropout = tf.placeholder_with_default(1.0, shape=())
        batch_size = tf.shape(self.input)[0]

        # Conv 1

        self.weights["conv1"] = tf.Variable(conf.conv1.weights)
        self.layers["conv1"] = conf.conv1.activation(tf.nn.conv2d(
            self.input,
            self.weights["conv1"],
            conf.conv1.strides,
            conf.conv1.padding, name="conv1"))
        self.weights["conv2"] = tf.Variable(conf.conv2.weights)
        self.layers["conv2"] = conf.conv2.activation(tf.nn.conv2d(
            self.layers["conv1"],
            self.weights["conv2"],
            conf.conv2.strides,
            conf.conv2.padding, name="conv2"))
        # flatten
        self.layers["flat"] = tf.reshape(self.layers["conv2"], (batch_size, -1))

        # fc1
        self.weights["fc1"] = tf.Variable(conf.fc1.weights)
        self.layers["fc1"] = conf.fc1.activation(tf.matmul(self.layers["flat"], self.weights["fc1"]))
        # fc2
        self.weights["fc2"] = tf.Variable(conf.fc2.weights)
        self.layers["fc2"] = conf.fc2.activation(tf.matmul(self.layers["fc1"], self.weights["fc2"]))
        self.layers["dropout"] = tf.nn.dropout(x=self.layers["fc2"], keep_prob=self.dropout)  # TODO

        # reshape
        shape = conf.reshape.shape
        shape[0] = batch_size
        self.layers["reshape"] = tf.reshape(self.layers["dropout"], shape)
        self.output = self.layers["reshape"]
        self.output_target = tf.placeholder(shape=self.output.shape, dtype=DTYPE)
        self.loss = conf_to_loss(conf.loss_type, self.output, self.output_target)
        optimizer = conf.optimizer
        with tf.control_dependencies([optimizer.minimize(self.loss)]):
            self.train_op = tf.identity(self.loss)


class SpecialNetwork1(Network):
    def __init__(self, input_shape, conf):
        self.weights = {}
        self.layers = Layers()
        self.input = tf.placeholder(shape=input_shape, dtype=DTYPE)
        batch_size = tf.shape(self.input)[0]
        # conv1
        self.weights["conv1"] = tf.Variable(conf.conv1.weights)
        self.layers["conv1"] = conf.conv1.activation(tf.nn.conv2d(
            self.input,
            self.weights["conv1"],
            conf.conv1.strides,
            conf.conv1.padding, name="conv1"))
        # conv2
        self.weights["conv2"] = tf.Variable(conf.conv2.weights)
        self.layers["conv2"] = conf.conv2.activation(tf.nn.conv2d(
            self.layers["conv1"],
            self.weights["conv2"],
            conf.conv2.strides,
            conf.conv2.padding, name="conv2"))
        # convT1
        output_shape = conf.convT1.output_shape
        output_shape[0] = batch_size
        self.weights["convT1"] = tf.Variable(conf.convT1.weights)
        self.layers["convT1"] = conf.convT1.activation(tf.nn.conv2d_transpose(
            self.layers["conv2"],
            self.weights["convT1"],
            output_shape,
            conf.convT1.strides,
            conf.convT1.padding, name="convT1"))
        # convT2
        output_shape = conf.convT2.output_shape
        output_shape[0] = batch_size
        self.weights["convT2"] = tf.Variable(conf.convT2.weights)
        self.layers["convT2"] = conf.convT2.activation(tf.nn.conv2d_transpose(
            self.layers["convT1"],
            self.weights["convT2"],
            output_shape,
            conf.convT2.strides,
            conf.convT2.padding, name="convT2"))
        self.output = self.layers["convT2"]
        self.output_target = tf.placeholder(shape=self.output.shape, dtype=DTYPE)
        self.loss = conf_to_loss(conf.loss_type, self.output, self.output_target)
        optimizer = conf.optimizer
        with tf.control_dependencies([optimizer.minimize(self.loss)]):
            self.train_op = tf.identity(self.loss)




class SpecialNetwork2(Network):
    def __init__(self, input_shape, conf):
        self.weights = {}
        self.layers = Layers()
        self.input = tf.placeholder(shape=input_shape, dtype=DTYPE)

        batch_size = tf.shape(self.input)[0]
        # conv1
        self.weights["conv1"] = tf.Variable(conf.conv1.weights)
        self.layers["conv1"] = conf.conv1.activation(tf.nn.conv2d(
            self.input,
            self.weights["conv1"],
            conf.conv1.strides,
            conf.conv1.padding, name="conv1"))
        # conv2
        self.weights["conv2"] = tf.Variable(conf.conv2.weights)
        self.layers["conv2"] = conf.conv2.activation(tf.nn.conv2d(
            self.layers["conv1"],
            self.weights["conv2"],
            conf.conv2.strides,
            conf.conv2.padding, name="conv2"))
        # flatten
        self.layers["flat"] = tf.reshape(self.layers["conv2"], (batch_size, -1))
        # fc1
        self.weights["fc1"] = tf.Variable(conf.fc1.weights)
        self.layers["fc1"] = conf.fc1.activation(tf.matmul(self.layers["flat"], self.weights["fc1"]))
        # fc2
        self.weights["fc2"] = tf.Variable(conf.fc2.weights)
        self.layers["fc2"] = conf.fc2.activation(tf.matmul(self.layers["fc1"], self.weights["fc2"]))
        # reshape
        shape = conf.reshape.shape
        shape[0] = batch_size
        self.layers["reshape"] = tf.reshape(self.layers["fc2"], shape)
        self.output = self.layers["reshape"]
        self.output_target = tf.placeholder(shape=self.output.shape, dtype=DTYPE)
        self.loss = conf_to_loss(conf.loss_type, self.output, self.output_target)
        optimizer = conf.optimizer
        with tf.control_dependencies([optimizer.minimize(self.loss)]):
            self.train_op = tf.identity(self.loss)
