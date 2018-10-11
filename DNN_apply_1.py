# /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf, pandas as pd, numpy as np, os
from sklearn import cross_validation as cv
import random
os.chdir("d:/tensorflow")

class current_data:
    """ 
    Initial a Dataset, and split it into train, test dataset, 
    also split features and targets based on parameters. 
    
    Args:
        file_path: file path of origin data, .csv file
        features_columns: features used in model
        targets_columns: labels for model
        split_ratio: between [0., 1.], define the ratio of test dataset
    """
    def __init__(self, file_path, features_colmuns, target_columns, split_ratio):
        self.dataset = pd.read_csv(file_path)
        self.features_columns = features_colmuns
        self.targets_columns = target_columns
        self.train_x, self.test_x, self.train_y, self.test_y = cv.train_test_split(self.dataset[self.features_columns], self.dataset[self.targets_columns], test_size=split_ratio, random_state=1)
        # train_test_split will not reset the index
        self.train_x = self.train_x.reset_index(drop=True)
        self.test_x = self.test_x.reset_index(drop=True).values
        self.train_y = self.train_y.reset_index(drop=True)
        self.test_y = self.test_y.reset_index(drop=True).values

    """
    Return a batch of dataset from train dataset

    Args:
        batch_size: sample number of each batch
    """    
    def next_batch(self, batch_size):
        total_sample_size = self.train_x.iloc[:,0].size
        # gernate sample with random, use this index to select sample data
        # the format is 2-D ndarray
        random_index = random.sample(range(total_sample_size),batch_size)
        return_batch = self.train_x.loc[random_index].reset_index(drop=True).values, self.train_y.loc[random_index].reset_index(drop=True).values
        return return_batch

"""
Create pandas.DataFrame from y_pred, y_real

Args:
    y_: real y
    y: predict of y by netural network model
    moderator: value to add when add operation
"""
def create_dataframe(y_, y, moderator):
    # predict value
    # result is a 2-D array, transform it to list, and reduce it by sum(v, [])
    y_pred = sum(sess.run(tf.exp(y) - moderator, feed_dict={x:c_dataset.test_x, y_:c_dataset.test_y}).tolist(), [])
    y_real = sum(sess.run(y_, feed_dict={y_:c_dataset.test_y}).tolist(), [])
    com_df = pd.DataFrame(np.array([y_pred, y_real]).T, columns=['p', 'r'])
    return com_df


file_path = "apple_features.csv"
features_colmuns = ['sku_offer_flag',
     'suit_offer_flag',
     'full_minus_offer_flag',
     'free_gift_flag',
     'ghost_offer_flag',
     'dq_and_jq_pay_flag',
     'non_promo_flag',
     'participation_rate_full_minus_and_suit_offer',
     'sku_offer_discount_rate',
     'full_minus_offer_discount_rate',
     'suit_offer_discount_rate',
     'ghost_offer_discount_rate',
     'dq_and_jq_pay_discount_rate',
     'free_gift_discount_rate',
     'newyear',
     'springfestival',
     'tombsweepingfestival',
     'labourday',
     'dragonboatfestival',
     'midautumnfestival',
     'nationalday',
     'h1111mark',
     'h618mark',
     'h1212mark',
     'day_of_week',
     'out_of_stock_flag',
     'week_of_year_fourier_sin_1',
     'week_of_year_fourier_cos_1',
     'week_of_year_fourier_sin_2',
     'week_of_year_fourier_cos_2',
     'week_of_year_fourier_sin_3',
     'week_of_year_fourier_cos_3',
     'day_of_week_fourier_sin_1',
     'day_of_week_fourier_cos_1',
     'day_of_week_fourier_sin_2',
     'day_of_week_fourier_cos_2',
     'day_of_week_fourier_sin_3',
     'day_of_week_fourier_cos_3',
     'sku_status_cd']
target_columns = ['sale_qtty']
split_ratio = 0.3
c_dataset = current_data(file_path, features_colmuns, target_columns, split_ratio)



input_size = len(features_colmuns)
output_size = len(target_columns)
learning_rate = 0.01
batch_size = 50
train_steps = 500
moderator = 1

x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
log_y_ = tf.log(y_ + moderator)

with tf.name_scope('parameters'):
    with tf.name_scope('weights'): 
        w = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(dtype=tf.float32, shape=[input_size, output_size], mean=0.0, stddev=0.1, seed=1))
        tf.summary.histogram('weight', w)
    with tf.name_scope('bias'):
        b = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(dtype=tf.float32, shape=[output_size], mean=0.0, stddev=0.1, seed=1))
        tf.summary.histogram('bias', b)

y = tf.matmul(x, w) + b
loss = tf.losses.mean_squared_error(labels=log_y_, predictions=y)
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
mse = tf.reduce_mean(tf.square(y-log_y_))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# merge all summary for tensorboard
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('logs/', sess.graph)
for i in range(train_steps):
    batch_x, batch_y = c_dataset.next_batch(batch_size)
    sess.run(train, feed_dict={x:batch_x, y_:batch_y})
    summary_writer.add_summary(sess.run(merged), global_step=i)
    print ("mse of test data：'%s'" % sess.run(mse, feed_dict={x:c_dataset.test_x, y_:c_dataset.test_y}))

com_df = create_dataframe(y_, y, moderator)
