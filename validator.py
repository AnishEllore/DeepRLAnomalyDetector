from constants import SHAPE
import matplotlib.pyplot as plt
import numpy as np
def validator(agent, env,episodes=1):
	for episode in range(episodes):
		current_state = env.reset()
		
		state_record = []
		action_record = []
		reward_record = []
		done = False
		while not done:
			reshaped_current_state = current_state.reshape(1,SHAPE[0],SHAPE[1])
			action = np.argmax(agent.get_qs(reshaped_current_state))
		

			new_state, reward, done = env.step(action)

		# Transform new continous state to new discrete state and count reward
			#episode_reward += reward

			current_state = new_state
			#step += 1
			state_record.append(current_state[len(current_state)-1,0])
			action_record.append(action)
			reward_record.append(reward)

		#plotting
		fig, axs = plt.subplots(3,sharex=True)
		fig.suptitle('Episode wise stats')
		axs[0].plot(state_record)
		axs[0].set_title('data')
		axs[1].plot(action_record)
		axs[1].set_title('action_record')
		axs[2].plot(reward_record)
		axs[2].set_title('reward_record')
		plt.show()
	
	print('validation done')
