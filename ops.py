import tensorflow as tf

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def conv2d(inputs, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, 
                            strides=strides, padding=padding, use_bias=use_bias)

def batch_norm(inputs, is_training=True, decay=0.9):
    return tf.contrib.layers.batch_norm(inputs, is_training=is_training, decay=decay)

