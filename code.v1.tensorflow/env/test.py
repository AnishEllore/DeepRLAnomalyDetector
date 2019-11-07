import pandas as pd
import numpy as np
import random
import os

import sklearn.preprocessing

NOT_ANOMALY = 0
ANOMALY = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]

# get all the path of the csv files to be loaded 
#repodir = '../env/time_series_repo/'
repodir = 'C:\\Users\\anish\\Desktop\\AnomalyDetection\\exp-anomaly-detector-master\\env\\time_series_repo'
repodirext = []

for subdir, dirs, files in os.walk(repodir):
	#print('bingo')
	for file in files:
		if file.find('.csv') != -1:
			repodirext.append(os.path.join(subdir, file))

print('cool')
print(repodirext)