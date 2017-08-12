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
import os
import tensorflow as tf       
import matplotlib.image as mpimg   


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))



###########################################################################
# initialize dataset
# need video processing                                       



filename = os.path.dirname("dataset/images/pushup/") + "/frame1.jpg";
image = mpimg.imread(filename) 
height, width, depth = image.shape 

training_shape = [width , height] 



X = tf.placeholder(tf.float32, [None, width * height]) # 320 * 240 = 76800 
Y1 = tf.placeholder(tf.float32, [None, 3]) # 3 output - need to be modified                    

###########################################################################








###########################################################################
# initialize weight
# initialize bias

#layer 1
W1 = tf.Variable(tf.truncated_normal(training_shape, stddev=0.05))
B1 = tf.Variable(tf.truncated_normal(training_shape, stddev=0.05))

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
        batch_xs, batch_ys = re
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  

###########################################################################


