# /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf, pandas as pd, numpy as np, os 
from tensorflow.examples.tutorials.mnist import input_data

os.chdir("D:/tensorflow")
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


in_units = 28*28
output_units = 10
learning_rate = 0.01
batch_size = 500
train_steps = 100

x = tf.placeholder(dtype=tf.float32, shape=[None, in_units], name='x_input')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, output_units], name='y_input')
w = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(shape=[in_units,output_units], mean=0, stddev=0.1, seed=1))
b = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(shape=[output_units], mean=0, stddev=0.1, seed=1))

output = tf.matmul(x,w) + b
#y = tf.matmul(x,w) + b
y = tf.nn.softmax(tf.matmul(x,w) + b)
# 交叉熵
# cross_entropy = tf.reduce_sum(y_*tf.log(y))
#cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output), axis=0)
cross_entropy = -tf.reduce_sum(y*y_)
# 准确性
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 梯度下降
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(train_steps):
    batch = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x:batch[0], y_:batch[1]})
    print (sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))





