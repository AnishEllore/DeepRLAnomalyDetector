LOOK_BACK = 5
FEATURES = 1
SHAPE = (LOOK_BACK,FEATURES+1)
FILE_PATH = "C:\\Users\\anish\\Desktop\\AnomalyDetection\\exp-anomaly-detector-master\\env\\time_series_repo\\A1Benchmark"

EPISODES = 1500
epsilon = 0.5  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 1000
MIN_REWARD = -200
MODEL_NAME = 'AnomalyDetection'