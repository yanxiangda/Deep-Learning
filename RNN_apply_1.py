# /usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf, numpy as np, pandas as pd, os
from sklearn import cross_validation as cv
import random
os.chdir("d:/tensorflow")


class current_data:
    """
    Generate hole dataset for rnn
    """
    def combine_dataset_rnn(self, df, height):
        data_np = df.values
        result_list = []
        for i in range(data_np.shape[0] - height + 1):
            result_list.append(data_np[i : (i + height)])
        return np.array(result_list)
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
        self.dataset = self.dataset.sort_values(by=['dt']).reset_index(drop=True)
        self.features_columns = features_colmuns
        self.targets_columns = target_columns
        self.train_x, self.test_x, self.train_y, self.test_y = cv.train_test_split(self.dataset[self.features_columns], self.dataset[self.targets_columns], test_size=split_ratio, random_state=1)
        # train_test1_split will not reset the index
        self.train_x = self.train_x.reset_index(drop=True)
        self.test_x = self.test_x.reset_index(drop=True).values
        self.train_y = self.train_y.reset_index(drop=True)
        self.test_y = self.test_y.reset_index(drop=True).values
        # make rnn sample
        self.dataset_rnn = self.dataset.iloc[0:700, :]
        self.dataset_rnn_out = self.dataset.iloc[700:, :]
        self.rnn_height = 7
        self.rnn_dataset_x = self.combine_dataset_rnn(self.dataset_rnn[self.features_columns], self.rnn_height)
        self.rnn_dataset_y = self.dataset_rnn[self.targets_columns].values[self.rnn_height-1:]
        self.train_x_rnn, self.test_x_rnn, self.train_y_rnn, self.test_y_rnn = cv.train_test_split(self.rnn_dataset_x, self.rnn_dataset_y, test_size=split_ratio, random_state=1)
        self.out_x_rnn = self.combine_dataset_rnn(self.dataset_rnn_out[self.features_columns], self.rnn_height)
        self.out_y_rnn = self.dataset_rnn_out[self.targets_columns].values[self.rnn_height-1:]
        
    """
    Return a batch of dataset from train dataset for dnn method

    Args:
        batch_size: sample number of each batch
    """    
    def next_batch_dnn(self, batch_size):
        total_sample_size = self.train_x.iloc[:,0].size
        # gernate sample with random, use this index to select sample data
        # the format is 2-D ndarray
        random_index = random.sample(range(total_sample_size),batch_size)
        return_batch = (self.train_x.loc[random_index].reset_index(drop=True).values, self.train_y.loc[random_index].reset_index(drop=True).values)
        return return_batch
    """
    Reture a batch of dataset from train dataset for rnn method
    """
    def next_batch_rnn(self, batch_size):
        total_sample_size = self.train_x_rnn.shape[0]
        random_index = random.sample(range(total_sample_size),batch_size)
        return_batch = (self.train_x_rnn[random_index], self.train_y_rnn[random_index])
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
    y_pred = sum(sess.run(tf.exp(y) - moderator, feed_dict={x:c_dataset.test_x_rnn, y_:c_dataset.test_y_rnn}).tolist(), [])
    y_real = sum(sess.run(y_, feed_dict={y_:c_dataset.test_y_rnn}).tolist(), [])
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

time_steps = 7
frame_size = len(features_colmuns)
output_size = len(target_columns)
learning_rate = 0.001
batch_size = 50
train_steps = 5000
hidden_size = 100
output_keep_prob = 0.6
moderator = 1

x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, frame_size])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
log_y_ = tf.log(y_ + moderator)

full_w = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(shape=[hidden_size, output_size], mean=0.0, stddev=0.1, dtype=tf.float32))
full_b = tf.Variable(dtype=tf.float32, initial_value=tf.constant(value=0.1, shape=[output_size]))

rnn_c2 = tf.nn.rnn_cell.BasicRNNCell(hidden_size,reuse=None)
okp = tf.placeholder(dtype=tf.float32)
rnn_c2 = tf.contrib.rnn.DropoutWrapper(cell=rnn_c2, input_keep_prob=1.0, output_keep_prob=okp)
h0 = rnn_c2.zero_state(batch_size, tf.float32)
outputs, status = tf.nn.dynamic_rnn(cell=rnn_c2, initial_state=h0, dtype=tf.float32, inputs=x)
full_output = tf.matmul(outputs[:,-1,:], full_w) + full_b
# 损失函数
loss = tf.losses.mean_squared_error(labels=log_y_, predictions=full_output)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

loss_list = []
for i in range(train_steps):
    batch_x, batch_y = c_dataset.next_batch_rnn(batch_size)
    sess.run(train, feed_dict={x:batch_x, y_:batch_y, okp:0.6})
    #print (sess.run(loss, feed_dict={x_:batch_x, y_:batch_y}))
    #print (sess.run(loss, feed_dict={x:batch_x, y_:batch_y}))
    current_loss = sess.run(loss, feed_dict={x:c_dataset.out_x_rnn[0:batch_size,:,:], y_:c_dataset.out_y_rnn[0:batch_size], okp:1.0})
    print (current_loss)
    loss_list.append(current_loss)

y_p = sess.run(full_output, feed_dict={x:c_dataset.test_x_rnn[0:batch_size,:,:], y_:c_dataset.test_y_rnn[0:batch_size], okp:1.0})
y_r = sess.run(log_y_, feed_dict={x:c_dataset.test_x_rnn[0:batch_size,:,:], y_:c_dataset.test_y_rnn[0:batch_size], okp:1.0})
com_df = pd.DataFrame(np.exp(np.array([y_p.reshape([-1]).tolist(), y_r.reshape([-1]).tolist()])).T-moderator, columns=['p', 'r'])

