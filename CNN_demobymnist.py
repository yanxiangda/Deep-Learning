# /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf, pandas as pd, numpy as np, os
from tensorflow.examples.tutorials.mnist import input_data

os.chdir("D:/tensorflow")
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

"""
Lenet-5 实际包括7层网络
卷积层-池化层-卷积层-池化层-全连接层-全连接层-softmax层
"""
input_size = 28*28
output_size = 10
# CNN如果学习速率过大，将会很难收敛
learning_rate = 0.001
train_steps = 100
batch_size = 50

x_ori = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
# tf.reshape 可以理解为将原来tensor的数据全部平铺，然后按照顺序添加到新的tensor中
x = tf.reshape(tensor=x_ori, shape=[-1, 28, 28, 1])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
w_layer_1 = tf.Variable(initial_value=tf.truncated_normal(dtype=tf.float32, mean=0.0, stddev=0.1, shape=[5,5,1,32]))
b_layer_1 = tf.Variable(initial_value=tf.constant(0.1, shape=[32]))
# 卷积层strides参数，[batch上移动，横向移动，纵向移动，channels移动]
# 第一层卷积之后，数据维度的变化为 batch_size*28*28*32
output_layer_1 = tf.nn.relu(tf.nn.conv2d(input=x, filter=w_layer_1, strides=[1,1,1,1], padding='SAME') + b_layer_1)
"""
池化层，
Args:
    ksize: ksize[0]=ksize[3]=1, 中间两个参数表示，池化层的大小
    strides: strides[0]=strides[3]=1, 中间两个参数表示池化步数的大小
Retures:
    第一次池化之后，数据维度的变化为 batch_size*14*14*32
"""
pool_layer_1 = tf.nn.max_pool(value=output_layer_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w_layer_2 = tf.Variable(initial_value=tf.truncated_normal(dtype=tf.float32, mean=0.0, stddev=0.1, shape=[5,5,32,64]))
b_layer_2 = tf.Variable(initial_value=tf.constant(0.1, shape=[64]))
# 第二层卷积之后，数据维度的变化为 batch_size*14*14*64
output_layer_2 = tf.nn.relu(tf.nn.conv2d(input=pool_layer_1, filter=w_layer_2, strides=[1,1,1,1], padding='SAME') + b_layer_2)
pool_layer_2 = tf.nn.max_pool(value=output_layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 第二次池化之后，数据维度的变化为 batch_size*7*7*64

"""
全连接层与softmax层
"""
w_layer_3 = tf.Variable(initial_value=tf.truncated_normal(dtype=tf.float32, mean=0.0, stddev=0.1, shape=[7*7*64, 1024]))
b_layer_3 = tf.Variable(initial_value=tf.constant(0.1, shape=[1024]))
output_layer_3 = tf.nn.relu(tf.matmul(tf.reshape(pool_layer_2, [-1, 7*7*64]), w_layer_3) + b_layer_3)
w_layer_4 = tf.Variable(initial_value=tf.truncated_normal(dtype=tf.float32, mean=0.0, stddev=0.1, shape=[1024, 10]))
b_layer_4 = tf.Variable(initial_value=tf.constant(0.1, shape=[10]))
output_layer_4 = tf.matmul(output_layer_3, w_layer_4) + b_layer_4
output_layer_4_softmax = tf.nn.softmax(output_layer_4)
#entropy = -tf.reduce_sum(y_ * tf.log(output_layer_4_softmax))
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_layer_4)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(output_layer_4_softmax, 1)), dtype=tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(train_steps):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x_ori:batch_x, y_:batch_y})
    print (sess.run(accuracy, feed_dict={x_ori:batch_x, y_:batch_y}))
    if i%50==0:
        print ("test accuracy: %s" % sess.run(accuracy, feed_dict={x_ori:mnist.test.images, y_:mnist.test.labels}))
sess.close()


