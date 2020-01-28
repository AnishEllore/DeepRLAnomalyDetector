from constants import SHAPE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate(tp, tn, fp, fn):
	precision = (tp + 1) / float(tp + fp + 1)
	recall = (tp + 1) / float(tp + fn + 1)
	f1 = 2 * ((precision * recall) / (precision + recall))
	print("Precision:{}, Recall:{}, F1-score:{} (f1 wrote to file)".format(precision, recall, f1))


def validator(agent, env, episodes=1,show=False):
	sns.set(style="whitegrid")
	TP_count = 0
	TN_count = 0
	FP_count = 0
	FN_count = 0

	for episode in range(episodes):
		current_state = env.reset()

		state_record = []
		action_record = []
		reward_record = []
		done = False
		while not done:
			reshaped_current_state = current_state.reshape(1, SHAPE[0], SHAPE[1])
			action = np.argmax(agent.get_qs(reshaped_current_state))
			new_state, reward, done = env.step(action, true_action=1, consider_agent=True)

			# Transform new continuous state to new discrete state and count reward
			# episode_reward += reward

			current_state = new_state
			# step += 1
			state_record.append(current_state[len(current_state) - 1, 0])
			action_record.append(action)
			reward_record.append(reward)
		TP_count_curr = reward_record.count(5)
		TN_count_curr = reward_record.count(1)
		FP_count_curr = reward_record.count(-1)
		FN_count_curr = reward_record.count(-5)
		TP_count += TP_count_curr
		TN_count += TN_count_curr
		FP_count += FP_count_curr
		FN_count += FN_count_curr
		print("Episode ", episode)
		evaluate(TP_count_curr, TN_count_curr, FP_count_curr, FN_count_curr)
		# plotting
		if show:
			fig, axs = plt.subplots(3, sharex=True)
			fig.suptitle('Episode wise stats')
			axs[0].plot(state_record)
			axs[0].set_title('data')
			axs[1].plot(action_record)
			axs[1].set_title('action_record')
			axs[2].plot(reward_record)
			axs[2].set_title('reward_record')
			# plt.savefig(f'results/fig{episode}.eps', format='eps', dpi=1200)
			plt.show()

	print('Overall')
	evaluate(TP_count, TN_count, FP_count, FN_count)
	print('validation done')
