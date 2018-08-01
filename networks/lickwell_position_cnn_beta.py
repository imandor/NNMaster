import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
import theano


class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        if not self._prediction:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._prediction = tf.nn.softmax(incoming)
        return self._prediction

    @property
    def optimize(self):
        if not self._optimize:
            cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize

    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error






def hidden_layer_output(model,X_input):
    model2 = keras.Sequential()
    model2.add(keras.layers.Convolution2D(256, kernel_size=(166, 5), activation='relu', padding='valid',
                                         input_shape=[X_input.shape[1], X_input.shape[2], X_input.shape[3]],weights=model.layers[0].get_weights()))
    model2.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2, padding='valid', data_format=None,weights=model.layers[1].get_weights()))
    model2.add(keras.layers.Convolution2D(128, kernel_size=(1, 5), activation='relu', padding='valid',
                                         input_shape=[X_input.shape[1], X_input.shape[2], X_input.shape[3]],weights=model.layers[2].get_weights()))
    model2.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2, padding='valid', data_format=None,weights=model.layers[3].get_weights()))
    model2.add(keras.layers.Flatten() )
    model2.add(keras.layers.Dense(6))
    activations = model2.predict(X_input)

    return activations

def cnn_model(shape, model_filename=None):

    if model_filename is None:
        print("Setting up model...")
        model = keras.Sequential()
        model.add(keras.layers.Convolution2D(256, kernel_size=(166, 5), activation='relu', padding='valid', input_shape=[shape[1], shape[2],shape[3]]))
        model.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2, padding='valid', data_format=None))
        model.add(keras.layers.Convolution2D(128, kernel_size=(1, 5), activation='relu', padding='valid', input_shape=[shape[1], shape[2], shape[3]]))
        model.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2, padding='valid', data_format=None))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(6))


        model.add(keras.layers.Activation('softmax'))
        model.compile(optimizer='SGD',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("Finished setting up model")
    else:
        print("Loading model...")
        model = load_model(model_filename)
        print("Finished loading model")
    return model