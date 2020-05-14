import tensorflow as tf
import tensorflow.contrib.slim as slim


def mlp_layer(d_in, d_out, activation, name):
    W = slim.variable(shape=(d_in, d_out), initializer=tf.contrib.layers.xavier_initializer(), name=name + '_W')
    b = slim.variable(shape=(1, d_out), initializer=tf.zeros_initializer, name=name + '_b')

    def apply_layer(x):
        a = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'softplus':
            return tf.nn.softplus(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a

    return apply_layer

