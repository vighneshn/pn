import tensorflow as tf


def weight_variable(shape, name):
    #initial = tf.truncated_normal(shape, stddev = 0.1)
    with tf.variable_scope(name):
        v = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    return v

def bias_variable(shape, name):
    #initial = tf.constant(0.1, shape = shape)
    with tf.variable_scope(name):
        v = tf.Variable(tf.constant(0.1, shape=shape))
    return v

def conv2D(x,W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def conv_layer(_input, shape, name):
    W = weight_variable(shape, name)
    b = bias_variable([shape[3]], name)
    return tf.nn.relu(conv2D(_input,W)+b)

def conv_layer_sigmoid(_input, shape, name):
    W = weight_variable(shape, name)
    b = bias_variable([shape[3]], name)
    return tf.nn.sigmoid(conv2D(_input,W)+b)

def conv_layer_lr(_input, shape, alpha, name):
    W = weight_variable(shape, name)
    b = bias_variable([shape[3]], name)
    return tf.nn.leaky_relu(conv2D(_input,W)+b, alpha)

def full_layer(_input, size, name):
    in_size = int(_input.get_shape()[1])
    W = weight_variable([in_size, size], name)
    b = bias_variable([size], name)
    return tf.matmul(_input,W)+b

