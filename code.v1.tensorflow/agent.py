import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import random, time
from constants import SHAPE,MODEL_NAME

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

class DQNagent:
	REPLAY_MEMORY_SIZE = 50_000
	MIN_REPLAY_MEMORY_SIZE = 1000
	MINIBATCH_SIZE = 64
	UPDATE_TARGET_EVERY = 10
	DISCOUNT = 0.99	
	def __init__(self,LOAD_MODEL=None,shape=SHAPE):
		self.LOAD_MODEL = LOAD_MODEL
		self.model = self.create_model(shape)
		self.target_model = self.create_model(shape)
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen = self.REPLAY_MEMORY_SIZE)

		self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
		self.target_update_counter = 0

	def create_model(self,shape):
		if self.LOAD_MODEL is not None:
			print(f"LOADING {self.LOAD_MODEL}")
			model = load_model(self.LOAD_MODEL)
			print(f"Model {self.LOAD_MODEL} loaded")
			return model
		else:
			model = Sequential()
			model.add(LSTM(64,input_shape = shape))
			model.add(Dense(2,activation='linear'))
			model.compile(loss='mae', optimizer='adam')
		return model

	def update_replay_memory(self, transition):
		# transition is (current_state, action, reward, new_state, done)
		self.replay_memory.append(transition)

	def get_qs(self,state):
		return self.model.predict(state)

	def train(self, terminal_state, step):
		if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory,self.MINIBATCH_SIZE)

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

		self.model.fit(current_states, np.array(Y),batch_size = self.MINIBATCH_SIZE, verbose=0, shuffle = False)

		# updating to determine if we want to update target_model 
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > self.UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0
