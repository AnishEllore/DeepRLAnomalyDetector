import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard
from keras import optimizers
import tensorflow as tf
from collections import deque
import random
import time
import keras
from constants import SHAPE, MODEL_NAME


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ReplayBuffer:
    def __init__(self, maxlen=50_000):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, transition):
        self.buffer.append(transition)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities)**priority_scale
        sample_probabilities = scaled_priorities/sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        normalized_importance = importance/max(importance)
        return normalized_importance

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = [self.buffer[index] for index in sample_indices]
        # samples = self.buffer[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices

    def set_priorities(self, indices, errors, offset=1.0):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def can_sample(self, min_replay_size):
        if len(self.buffer) < min_replay_size:
            return False
        return True


class DQNagent:
    REPLAY_MEMORY_SIZE = 50_000
    MIN_REPLAY_MEMORY_SIZE = 1000
    MINI_BATCH_SIZE = 64
    UPDATE_TARGET_EVERy = 10
    DISCOUNT = 0.99

    def __init__(self, trained_model=None, shape=SHAPE):
        self.TRAINED_MODEL = trained_model
        self.model = self.create_model(shape)
        self.target_model = self.create_model(shape)
        self.target_model.set_weights(self.model.get_weights())

        # self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.replay_memory = ReplayBuffer()
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def create_model(self, shape):
        if self.TRAINED_MODEL is not None:
            print(f"LOADING {self.TRAINED_MODEL}")
            model = load_model(self.TRAINED_MODEL)
            print(f"Model {self.TRAINED_MODEL} loaded")
            return model
        else:
            model = Sequential()
            model.add(LSTM(32, input_shape=shape))
            # model.add(Dropout(0.2))
            # model.add(BatchNormalization())
            # model.add(Dense(32, activation='relu'))
            # model.add(Dropout(0.2))
            model.add(Dense(2, activation='linear'))
            model.compile(loss='mse', optimizer='adam')

        return model

    def update_replay_memory(self, transition):
        # transition is (current_state, action, reward, new_state, done)
        self.replay_memory.add(transition)

    def get_qs(self, state):
        return self.model.predict(state)

    def train(self, terminal_state,epsilon):
        if not self.replay_memory.can_sample(self.MIN_REPLAY_MEMORY_SIZE):
            return
        # mini_batch = random.sample(self.replay_memory, self.MINI_BATCH_SIZE)
        mini_batch, importance, indices = self.replay_memory.sample(self.MINI_BATCH_SIZE, 0.7)
        current_states = np.array([transition[0] for transition in mini_batch])
        # print(current_states)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []
        errors = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            errors.append(new_q-current_qs[action])
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)
        '''
        Annealing the bias for whole batch instead of each sample
        
        to be done
        '''
        annealing_bias = sum(importance**(1-epsilon))/len(importance)
        # self.model.lr.set_values(0.001*annealing_bias)

        # opt = keras.optimizers.Adam(lr=0.001*annealing_bias, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # self.model.compile(loss='mse', optimizer=opt)
        # history = LossHistory()
        self.model.fit(current_states, np.array(y), batch_size=self.MINI_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        '''
        one method of updating priorities
        1.) losses
        2.) initial errors before training
        '''
        self.replay_memory.set_priorities(indices, errors)
        # print(history.losses)
        # updating to determine if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > self.UPDATE_TARGET_EVERy:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
