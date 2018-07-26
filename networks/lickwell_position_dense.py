import tensorflow as tf

import numpy as np


def lickwell_position_dense_model_fn(features, labels,mode):
    # Input

    input_layer = tf.reshape(features["x"],  [-1, features["x"].shape[1],features["x"].shape[2], features["x"].shape[3]])

    # Dense layer 1

    dense1 = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.nn.relu)

    #  Dropout layer 1

    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense layer 2

    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)

    #  Dropout layer 2

    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense layer 2

    dense3 = tf.layers.dense(inputs=dropout2, units=1024, activation=tf.nn.relu)

    #  Dropout layer 2

    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten

    dropout_flat = tf.reshape(dropout3, [-1, dropout3.shape[1] * dropout3.shape[2] * dropout3.shape[3]])

    # Logits layer

    logits = tf.layers.dense(inputs=dropout_flat,units=6)

    # Generate predictions for PREDICT and EVAL

    predictions = {
        "classes": tf.argmax(input=logits,axis=1),
        "probabilities": tf.nn.softmax(logits,name="softmax_tensor")
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
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


