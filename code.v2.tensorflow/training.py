from constants import FILE_PATH
from environment import DataEnvironment
from agent import DQNagent
from validator import validator
import random
import numpy as np
import os

random.seed(1)
np.random.seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')
env = DataEnvironment(FILE_PATH)
LOAD_MODEL = 'models/AnomalyDetection___951.00max__797.20avg__607.00min__1572649628.model'
agent = DQNagent(LOAD_MODEL)

'''
ep_rewards = [-200]
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

	# Update tensorboard step every episode
	#agent.tensorboard.step = episode

	# Restarting episode - reset episode reward and step number
	episode_reward = 0
	step = 1

	# Reset environment and get initial state
	current_state = env.reset()

	# Reset flag and start iterating until episode ends
	done = False
	cnt=0
	while not done:
		cnt+=1
		# This part stays mostly the same, the change is to query a model for Q values
		if np.random.random() > epsilon:
			# Get action from Q table
			reshaped_current_state = current_state.reshape(1,SHAPE[0],SHAPE[1])
			#print(current_state)
			action = np.argmax(agent.get_qs(reshaped_current_state))
		else:
			# Get random action
			action = np.random.randint(0, env.ACTION_SPACE_SIZE)

		new_state, reward, done = env.step(action)

		# Transform new continous state to new discrete state and count reward
		episode_reward += reward

		#if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
		#if SHOW_PREVIEW and not episode % 1:	
			#env.render()

		# Every step we update replay memory and train main network
		agent.update_replay_memory((current_state, action, reward, new_state, done))
		agent.train(done, step)

		current_state = new_state
		step += 1
	#print(cnt)
	# Append episode reward to a list and log stats (every given number of episodes)
	ep_rewards.append(episode_reward)
	if not episode % AGGREGATE_STATS_EVERY or episode == 1:
		average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
		min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
		max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
		#agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
		#print(average_reward,min_reward,max_reward)
		# Save model, but only when min reward is greater or equal a set value
		#if min_reward >= MIN_REWARD:
			#agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

	if episode >= 10:
    	agent.model.save(f'models/{MODEL_NAME}_EPISODES_{EPISODES}{int(time.time())}.model')

	# Decay epsilon
	if epsilon > MIN_EPSILON:
		epsilon *= EPSILON_DECAY
		epsilon = max(MIN_EPSILON, epsilon)


print('done')
'''
validator(agent,env,10)

