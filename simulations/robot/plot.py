import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os, sys

def get_reward(xi, theta):
		laptop = np.array([0.35, 0.1, 0.0])
		context = np.array([0.7, -0.45, 0.1])
		R1 = 0
		R2 = 0
		for idx in range(len(xi)):
			R1 -= np.linalg.norm(xi[idx, :2] - laptop[:2])
			R2 -= np.linalg.norm(context - xi[idx])
		R1 *= theta[0]
		R2 *= theta[1]

		return R1 + R2

# noise_arr = np.array([0.0, 0.025, 0.05, 0.1])
# bias_arr = np.array([0.0, 0.025, 0.05, -0.025, -0.05])

# fig, ax = plt.subplots(len(noise_arr), len(bias_arr), figsize=(15,10))

# for noise_id, noise in enumerate(noise_arr):
# 	for bias_id, bias in enumerate(bias_arr):
# 		save_name = 'results/test_1/noise_{}_bias_{}/'.format(noise, bias)

# 		algo = ['ours', 'b1', 'b2', 'b3']

# 		for alg_id, alg in enumerate(algo):
# 			data = pickle.load(open(save_name + 'data_' +  alg + '.pkl', 'rb'))

# 			regret = []

# 			xi = data['traj_learned']
# 			xi_ideal = data['traj_ideal']

# 			for idx in range(len(xi)):

# 				reward_ideal = get_reward(xi_ideal[idx], data['theta_star'][idx])
# 				reward = get_reward(xi[idx], data['theta_star'][idx])

# 				regret.append(reward_ideal - reward)

		
# 			ax[noise_id, bias_id].bar(alg_id, np.mean(regret), yerr=np.std(regret)/np.sqrt(len(regret)))
# 		ax[noise_id, bias_id].set_title('noise: {}, bias: {}'.format(noise, bias))
# 		ax[noise_id, bias_id].set_xticks(np.arange(len(algo)))
# 		ax[noise_id, bias_id].set_xticklabels(algo)
# fig.tight_layout()
# plt.show()



p = argparse.ArgumentParser()
p.add_argument('--noise', type=float, default=0.025)
p.add_argument('--bias', type=float, default=0.0)
args = p.parse_args()

noise = args.noise
bias = args.bias

save_name = 'results/rebuttal/ch_pref_noise_{}_bias_{}'.format(noise, bias)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)

# algo = ['ours', 'e2e', 'b1', 'b2','b3']
algo = ['b3', 'b1', 'b2', 'e2e', 'ours']

for alg_id, alg in enumerate(algo):
	data = pickle.load(open(save_name + '/data_' + alg + '.pkl', 'rb'))
	
	regret = []

	xi = data['traj_learned']
	xi_ideal = data['traj_ideal']

	for idx in range(len(xi)):

		reward_ideal = get_reward(xi_ideal[idx], -1*data['theta_star'][idx])
		reward = get_reward(xi[idx], -1*data['theta_star'][idx])

		regret.append(reward_ideal - reward)
	print(alg)
	print(regret)
	input()
	os.system('clear')
	
	ax1.bar(alg_id, np.mean(regret), yerr=np.std(regret)/np.sqrt(len(regret)))
	ax2.bar(alg_id, np.mean(regret), yerr=np.std(regret)/np.sqrt(len(regret)))
# exit()
ax1.set_ylim(2.0, 3.25)
ax2.set_ylim(0, 0.75)
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax2.set_xticks(np.arange(len(algo)))
ax2.set_xticklabels(algo)
ax1.set_title('regret (noise = ' + str(args.noise) + ', bias = ' + str(args.bias) + ')')

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.savefig('results/rebuttal/ch_pref_noise_{}_bias_{}.svg'.format(noise, bias))
plt.savefig('results/rebuttal/ch_pref_noise_{}_bias_{}.png'.format(noise, bias))
plt.show()