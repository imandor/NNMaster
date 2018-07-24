import tensorflow as tf

import numpy as np



def lick_cnn_model(X,mode=False):
    # Input
    input_layer = tf.reshape(X,[-1,X.shape[1],X.shape[2],X.shape[3]])#tf.reshape(features["x"],  [-1, 166,38, 1])

    # Convolutional layer 1

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=256, kernel_size=[166, 5], padding="valid",
                             activation=tf.nn.relu)

    # Pooling layer 1

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)

    # Convolutional layer 2

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[1, 5], padding="valid", activation=tf.nn.relu)

    # Pooling layer 2

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

    # Dense layer

    pool2_flat = tf.reshape(pool2, [-1, 64])  # T # 166->41, 84->21, 136-> 24
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer

    logits = tf.layers.dense(inputs=dropout, units=6)

    return logits