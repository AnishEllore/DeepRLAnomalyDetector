LOOK_BACK = 5
FEATURES = 1
SHAPE = (LOOK_BACK, FEATURES+1)
FILE_PATH = "env/time_series_repo/A1Benchmark"

EPISODES = 5000
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 500
MIN_REWARD = 0
MODEL_NAME = 'AnomalyDetector_pass1'
