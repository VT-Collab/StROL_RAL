import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse 
import os, sys


# noise_arr = np.array([0.0, 0.05, 0.1, 0.25])
# bias_arr = np.array([0.0, 0.05, 0.1, 0.25])

# fig, ax = plt.subplots(len(noise_arr), len(bias_arr), figsize=(15,10))


# for noise_id, noise in enumerate(noise_arr):
#     for bias_id, bias in enumerate(bias_arr):
#         save_name = 'g_func_exp/results/results_latest/noise_{}_bias_{}/'.format(noise, bias)
#         algo = ['ours', 'e2e', 'b1', 'b2', 'b3']

#         for alg_id, alg in enumerate(algo):
#             data = pickle.load(open(save_name + alg + '.pkl', 'rb'))

#             theta_star = np.array(data['theta_star'])
#             for t_idx, ts in enumerate(theta_star):
#                 theta_star[t_idx] = ts/np.linalg.norm(ts)
#             theta_pred = np.array(data['theta_predicted'])

#             err_mean = np.mean(np.linalg.norm(theta_star - theta_pred, axis=1))

#             err_std = np.std(np.linalg.norm(theta_star - theta_pred, axis=1))

#             ax[noise_id, bias_id].bar(alg_id, err_mean, yerr=err_std/np.sqrt(250))
#         ax[noise_id, bias_id].set_title('noise: {}, bias: {}'.format(noise, bias))
#         ax[noise_id, bias_id].set_xticks(np.arange(len(algo)))
#         ax[noise_id, bias_id].set_xticklabels(algo)
# fig.tight_layout()
# fig.savefig('g_func_exp/results/err_theta.png')
# fig.savefig('g_func_exp/results/err_theta.svg')
# plt.show()


p = argparse.ArgumentParser()
p.add_argument('--noise', type=float, default=0.0)
p.add_argument('--bias', type=float, default=0.0)
args = p.parse_args()

noise = args.noise
bias = args.bias

fig, ax = plt.subplots()
save_name = 'g_func_exp/results/results_latest/noise_{}_bias_{}_uniform/'.format(noise, bias)

# algo = ['ours', 'e2e', 'b1', 'b2', 'b3']
algo = ['b1', 'b2', 'b3', 'e2e', 'ours']
ERROR = {'b1': [], 'b2': [], 'b3': [], 'e2e': [], 'ours': []}
for alg_id, alg in enumerate(algo):
    data = pickle.load(open(save_name + alg + '.pkl', 'rb'))

    theta_star = np.array(data['theta_star'])
    for t_idx, ts in enumerate(theta_star):
        theta_star[t_idx] = ts/np.linalg.norm(ts)
    theta_pred = np.array(data['theta_predicted'])

    ERROR[alg].append(np.linalg.norm(theta_star - theta_pred, axis=1))
    err_mean = np.mean(np.linalg.norm(theta_star - theta_pred, axis=1))

    err_std = np.std(np.linalg.norm(theta_star - theta_pred, axis=1))

    ax.bar(alg_id, err_mean, yerr=err_std/np.sqrt(250))
    print(alg, len(ERROR[alg][-1]))
    print(*ERROR[alg][-1])
    input()
    os.system('clear')


ax.set_title('noise: {}, bias: {}'.format(noise, bias))
ax.set_xticks(np.arange(len(algo)))
ax.set_xticklabels(algo)
# plt.savefig('g_func_exp/results/noise_{}_bias_{}_uniform.svg'.format(noise, bias))
# plt.show()  




