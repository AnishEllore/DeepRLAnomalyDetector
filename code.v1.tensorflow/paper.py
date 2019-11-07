import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import sys
import os
import time

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import rnn
#from tensorflow import rnn
from mpl_toolkits.mplot3d import axes3d

from collections import deque, namedtuple

# import the library in the sub-folder env
from env.time_series_repo_ext import EnvTimeSeriesfromRepo

# macros for running q-learning
DATAFIXED = 0           # whether target at a single time series dataset

EPISODES = 500          # number of episodes for training
DISCOUNT_FACTOR = 0.5   # reward discount factor [0,1]
EPSILON = 0.5           # epsilon-greedy method parameter for action selection
EPSILON_DECAY = 1.00    # epsilon-greedy method decay parameter

NOT_ANOMALY = 0
ANOMALY = 1

action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25        # size of the slide window for SLIDE_WINDOW state and reward functions
n_input_dim = 2     # dimension of the input for a LSTM cell
n_hidden_dim = 64   # dimension of the hidden state in LSTM cell

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5


# The state function returns a vector composing of n_steps of n_input_dim data instances:
# e.g., [[x_1, f_1], [x_2, f_2], ..., [x_t, f_t]] of shape (n_steps, n_input_dim)
# x_t is the new data instance. t here is equal to n_steps

def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    if timeseries_curser == n_steps:
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])

        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])

        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        state0 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 1]]))

        return np.array([state0, state1], dtype='float32')


# Also, because we use binary tree here, the reward function returns a list of rewards for each action
def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0):
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]

        if timeseries['anomaly'][timeseries_curser] == 1:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]