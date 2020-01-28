LOOK_BACK = 5
FEATURES = 1
SHAPE = (LOOK_BACK, FEATURES+1)
FILE_PATH = "env/time_series_repo/A1Benchmark"

EPISODES = 134
epsilon = 0.9  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 67
MIN_REWARD = 0
MODEL_NAME = 'testA1(134epochs)'
