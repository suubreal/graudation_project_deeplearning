# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 17:01:00 2017

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:10:47 2017

@author: USER
"""
# import tensorflow
import tensorflow as tf          



###########################################################################
# initialize dataset
# need video processing 
from tensorflow.examples.tutorials.mnist import input_data                              
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)            

X = tf.placeholder(tf.float32, [None, 76800]) # 320 * 240 = 76800 
#Y1 = tf.placeholder(tf.float32, [None, 3]) # 3 output - need to be modified                    

###########################################################################








###########################################################################
# initialize weight
# initialize bias

#layer 1
W1 = tf.Variable()
B1 = tf.Variable()

#Layer 2
#W2 = tf.Variable()
#B2 = tf.Variable()


###########################################################################







###########################################################################
# model           
#prediction = tf.nn.softmax(tf.matmul(x, W) + b)    

#layer 1
Y1 = tf.nn.relu(tf.nn.conv2d(X , W1 , strides = [1 , 1 , 1 , 1] , padding='SAME') + B1)

#Layer 2
#Y2 = tf.nn.relu(tf.nn.conv2d(Y1  W2 , strides = [1 , 2 , 2 , 1] , padding='SAME') + B2)


Y = tf.n.softmax(Y1)

###########################################################################
           




###########################################################################
# training

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(Y1), reduction_indices=[1]))     
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)   
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y1, 1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        

for i in range(1000):                           
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  

###########################################################################


