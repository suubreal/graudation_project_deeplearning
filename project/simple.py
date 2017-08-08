# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:10:47 2017

@author: USER
"""

import tensorflow as tf          

from tensorflow.examples.tutorials.mnist import input_data                              
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)             #  download and read in the MNIST data

x = tf.placeholder(tf.float32, [None, 784]) 
y = tf.placeholder(tf.float32, [None, 10])                            # We will input a value for x and y when we ask TensorFlow to run a computation. We want to be able to input any number of MNIST images, each flattened into
                                                                                           #  a 784-dimensional vector. We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784].
                                                                                            # (Here None means that a dimension can be of any length) 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))                                              # A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. We use Variables to define the weights and biases

prediction = tf.nn.softmax(tf.matmul(x, W) + b)                 # implement our model

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))     # cross-entropy is measuring how inefficient our predictions are for describing the truth. The cross-entropy is defined as the loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)     # we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))    # One last thing before we launch it, we have to create an operation to initialize the variables we created
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        # calculate the accuracy of prediction 

for i in range(1000):                            # We can now launch the model in a Session, and train it 1000 times
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  

