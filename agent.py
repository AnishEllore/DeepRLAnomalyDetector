import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import random
from constants import SHAPE


class DQNagent:
    REPLAY_MEMORY_SIZE = 50_000
    MIN_REPLAY_MEMORY_SIZE = 1000
    MINIBATCH_SIZE = 32
    UPDATE_TARGET_EVERY = 5
    DISCOUNT = 0.99

    def __init__(self, LOAD_MODEL=None, shape=SHAPE):
        self.LOAD_MODEL = LOAD_MODEL
        self.model = self.create_model(shape)
        self.target_model = self.create_model(shape)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{MODEL_NAME}{int(time.time())}", histogram_freq=1)
        self.target_update_counter = 0

    def create_model(self, shape):
        if self.LOAD_MODEL is not None:
            print(f"LOADING {self.LOAD_MODEL}")
            model = load_model(self.LOAD_MODEL)
            print(f"Model {self.LOAD_MODEL} loaded")
            return model
        else:
            model = Sequential()
            model.add(LSTM(32, input_shape=shape))
            model.add(Dense(2, activation='linear'))
            model.compile(loss='mae', optimizer='adam')
            # model = Sequential()
            # model.add(LSTM(32,input_shape = shape), return_sequences=True)
			# model.add(Dropout(0.2))
			# model.add(BatchNormalization())
			# model.add(LSTM(32, input_shape=shape)
			# model.add(Dropout(0.2))
			# model.add(BatchNormalization())
			# model.add(Dense(32, activation = "relu"))
			# model.add(Dropout(0.2))
			# model.add(Dense(2,activation='linear'))
            # opt = SGD(lr=0.01, momentum=0.9)
            # model.compile(loss = 'binary_crossentropy', optimizer=opt)
            return model

    def update_replay_memory(self, transition):
        # transition is (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(state)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(current_states, np.array(Y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False)

        # updating to determine if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
