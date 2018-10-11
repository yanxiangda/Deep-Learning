#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf, pandas as pd, numpy as np, os
from tensorflow.examples.tutorials.mnist import input_data
os.chdir("D:/tensorflow")
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

sequence_size = 28
frame_size = 28
output_size = 10
batch_size = 50
hidden_size = 128
train_steps = 100
learning_rate=0.001


x_ = tf.placeholder(dtype=tf.float32, shape=[None, sequence_size*frame_size])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
x = tf.reshape(x_, [-1, sequence_size, frame_size])

full_w = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(shape=[hidden_size, output_size], mean=0.0, stddev=0.1, dtype=tf.float32))
full_b = tf.Variable(dtype=tf.float32, initial_value=tf.constant(value=0.1, shape=[output_size]))


rnn_c2 = tf.nn.rnn_cell.BasicRNNCell(hidden_size,reuse=None)
h0 = rnn_c2.zero_state(batch_size, tf.float32)
outputs, status = tf.nn.dynamic_rnn(cell=rnn_c2, initial_state=h0, dtype=tf.float32, inputs=x)
full_output = tf.matmul(outputs[:,-1,:], full_w) + full_b
# 损失函数
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=full_output)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(full_output, 1)), tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(500):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x_:batch_x, y_:batch_y})
    #print (sess.run(loss, feed_dict={x_:batch_x, y_:batch_y}))
    print (sess.run(accuracy, feed_dict={x_:batch_x, y_:batch_y}))

sess.run(tf.argmax(tf.nn.softmax(full_output), 1), feed_dict={x_:batch_x, y_:batch_y})
sess.run(tf.argmax(y_, 1), feed_dict={x_:batch_x, y_:batch_y})








