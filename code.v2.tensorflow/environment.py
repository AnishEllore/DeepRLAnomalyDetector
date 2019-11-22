import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from constants import LOOK_BACK
class DataEnvironment:
	ANOMALY = 1
	NOT_ANOMALY = 0
	ACTION_SPACE = [NOT_ANOMALY, ANOMALY]
	ACTION_SPACE_SIZE = len(ACTION_SPACE)
	TP_REWARD = 10
	TN_REWARD = 1
	FP_REWARD = -1
	FN_REWARD = -10

	def __init__(self,repo_dir,look_back=LOOK_BACK):
		self.repo_dir = repo_dir
		self.repo_files = []
		self.data = []
		self.target = []
		self.current_file_id = 0
		self.fix_dataset = False
		self.look_back = look_back
		for subdir, dirs, files in os.walk(self.repo_dir):
			for file in files:
				if file.find('.csv') != -1:
					self.repo_files.append(os.path.join(subdir, file))

		self.repo_data, self.repo_target_data = self.files_data_loading(self.repo_files)


	def reset(self):
		if not self.fix_dataset:
			self.current_file_id = (self.current_file_id + 1)%len(self.repo_files)
		print(self.current_file_id)
		self.data, self.target = self.repo_data[self.current_file_id], self.repo_target_data[self.current_file_id]
		self.index_data = self.look_back
		self.state = self.state_function(self.data, self.index_data)
		
		return self.state

	def step(self,action):
		reward = self.reward_function(action)
		self.index_data+=1
		if self.index_data >= len(self.data):
			done=1
		else:
			done=0
			self.state = self.state_function(self.data, self.index_data, self.state, action)

		return self.state, reward, done

	def files_data_loading(self, repo_files):
		repo_data = []
		repo_target_data = []
		for file_name in repo_files:
			df = pd.read_csv(file_name, usecols=[1,2], header=0, names=['value','anomaly'])
			values = df[['value']].values.astype(np.float32)
			targets = df[['anomaly']].values.astype(np.float32)
			scaler = MinMaxScaler()
			scaled_values = scaler.fit_transform(values)
			repo_data.append(scaled_values)
			
			scaled_targets = scaler.fit_transform(targets)
			repo_target_data.append(scaled_targets)

		return repo_data, repo_target_data

	def reward_function(self,action):
		
		if self.target[self.index_data] == 1:
			if action==1:
				return self.TP_REWARD
			else:
				return self.FN_REWARD
		else:
			if action==0:
				return self.TN_REWARD
			else:
				return self.FP_REWARD
		

	def state_function(self,timeseries, timeseries_curser, previous_state=np.array([]), action=0):
		if timeseries_curser == self.look_back:
			zeros = np.zeros((self.look_back,1))
			state = timeseries[:self.look_back,:]
			state = np.append(state,zeros,axis=1)
			return state
		
		if timeseries_curser > self.look_back:
			state_time = timeseries[timeseries_curser,:]
			state_time = np.append(state_time,action).reshape((1,-1))
			state = np.append(previous_state[1:,:

				],state_time,axis=0)
			return state

