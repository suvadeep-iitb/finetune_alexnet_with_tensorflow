"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, exp, num_classes, emb_dim, c=0.005, nel = 0):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.EMB_DIM = emb_dim
        self.EXP = exp
        self.C = c
        self.NEL = nel

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""

        fc6 = fc(self.X, 6*6*256, self.EMB_DIM, name='fc6')
        if self.NEL >= 4:
            fc6 = eexponentiation(fc6, self.EXP, self.C)
        dropout6 = dropout(fc6, self.KEEP_PROB)

        fc7 = fc(dropout6, self.EMB_DIM, self.EMB_DIM, name='fc7')
        if self.NEL >= 3:
            fc7 = eexponentiation(fc7, self.EXP, self.C)
        dropout7 = dropout(fc7, self.KEEP_PROB)

        fc8 = fc(dropout7, self.EMB_DIM, self.EMB_DIM, name='fc8')
        if self.NEL >= 2:
            fc8 = eexponentiation(fc8, self.EXP, self.C)
        dropout8 = dropout(fc8, self.KEEP_PROB)

        self.fc9 = fc(dropout8, self.EMB_DIM, self.NUM_CLASSES, relu=False, name='fc9')
        if self.NEL >= 1:
            self.fc9 = eexponentiation(self.fc9, self.EXP, self.C)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def eexponentiation(x, e, c):
    ndims = len(x.get_shape().as_list())
    exp = tf.tile(tf.reshape(e, [-1]+[1]*(ndims-1)), 
                  tf.slice(tf.shape(tf.expand_dims(x, 1)), [1], [ndims]))
    if c > 0.0:
        ly = tf.pow(tf.maximum(c, tf.abs(x)), exp) - tf.pow(c, exp)
        sy = tf.minimum(c, tf.abs(x)) * tf.pow(c, exp-1)
        output = tf.multiply(tf.sign(x), ly+sy)
    else:
        output = tf.multiply(tf.sign(x), tf.pow(tf.abs(x), exp))
    return output


