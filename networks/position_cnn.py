import tensorflow as tf

import numpy as np


def cnn_model_fn(features, labels,mode):
    # Input
    input_layer = tf.reshape(features["x"],  [-1, features["x"].shape[1],features["x"].shape[2], features["x"].shape[3]])


    # Convolutional layer 1

    conv1 = tf.layers.conv2d(inputs = input_layer,filters = 256, kernel_size=[166,5],padding="valid", activation=tf.nn.relu)

    # Pooling layer 1

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[1, 2], strides=2)

    # Convolutional layer 2

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size=[1,5], padding="valid",activation=tf.nn.relu)

    # Pooling layer 2

    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[1,2],strides=2)

    # Dense layer

    pool2_flat = tf.reshape(pool2,[-1,384]) # T # 166->41, 84->21, 136-> 24
    dense = tf.layers.dense(inputs=pool2_flat,units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,rate=0.4,training=mode ==tf.estimator.ModeKeys.TRAIN)

    # Logits layer

    logits = tf.reshape(tf.layers.dense(inputs=dropout,units=1),[-1,])

    # Generate predictions for PREDICT and EVAL

    predictions = {
        "classes": logits,
        "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    # Calculate Loss(TRAIN,EVAL)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure Training (TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (EVAL)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    # sess = tf.InteractiveSession()
    # with sess.as_default():
    #     tf.initialize_all_variables().run()
    #     print("logits:",logits.eval())
    #     print("asd")
    tensors_to_log = {"probabilities": logits}
    logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                               "accuracy": logits}, every_n_iter=50)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops,training_hooks=logging_hook)


