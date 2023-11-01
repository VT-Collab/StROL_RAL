import numpy  as np
import pickle
import matplotlib.pyplot as plt

def reward(traj, theta_star, idt):
    phi = np.zeros(3)
    if idt != 0:
        phi = np.zeros(4)
    laptop = np.array([0.35, 0.1, 0.0])
    goal = np.array([0.7, -0.45, 0.1])
    orienration = np.array([0.0, 0.0, 0.0])
    for idx in range(len(phi)):
        if idx == 0:
            for s_idx in range(len(traj)):
                dist = np.linalg.norm(laptop[:2] - traj[s_idx, :2]) 
                phi[idx] -= dist
        if idx == 1:
            for s_idx in range(len(traj)):
                phi[idx] -= np.linalg.norm(goal[:3] - traj[s_idx, :3])
        if idx == 2:
            phi[idx] -= np.linalg.norm(traj[s_idx, 2])
        if idx == 3:
            phi[idx] -= np.linalg.norm(orienration - traj[s_idx, 3:])

    reward = theta_star @ phi
    return reward


# users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_8', 'user_9', 'user_10', 'user_11', 'user_12']
# corr_num = {'b1': 0, 'b2': 0, 'ours': 0}
# for idx, user in enumerate(users):
#     tasks = ['task_1', 'task_2', 'task_3']
#     for idt, task in enumerate(tasks):
#         algo = ['b1', 'b2', 'ours']
#         for ida, alg in enumerate(algo):
#             data = pickle.load(open('data/{}/{}/{}/data.pkl'.format(user, task, alg), 'rb'))
#             corr = data['corrections'][-1]
#             for c in corr:
#                 corr_num[alg] += len(c)


# print(corr_num)

"""PLOT CORRECTION NUMBERS"""

users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_8', 'user_9', 'user_10', 'user_11', 'user_12']
corr_num = {'b1': [], 'b2': [], 'ours': []}
for idx, user in enumerate(users):
    tasks = ['task_1', 'task_2', 'task_3']
    for idt, task in enumerate(tasks):
        algo = ['b1', 'b2', 'ours']
        for ida, alg in enumerate(algo):
            data = pickle.load(open('data/{}/{}/{}/data.pkl'.format(user, task, alg), 'rb'))
            corr = data['corrections'][-1]
            num = 0
            for c in corr:
                num += len(c)
            corr_num[alg].append(num)
corr_num['b1'] = np.array(corr_num['b1'])
corr_num['b2'] = np.array(corr_num['b2'])
corr_num['ours'] = np.array(corr_num['ours'])
print(np.mean(0.05*corr_num['b1']), np.std(0.05*corr_num['b1']))
print(np.mean(0.05*corr_num['b2']), np.std(0.05*corr_num['b2']))
print(np.mean(0.05*corr_num['ours']), np.std(0.05*corr_num['ours']))
# print(corr_num)
# print(np.sum(corr_num['b1']))
# print(np.sum(corr_num['b2']))
# print(np.sum(corr_num['ours']))
fig, ax = plt.subplots()

ax.bar(0, np.mean(corr_num['b1']), yerr=np.std(corr_num['b1'])/np.sqrt(len(corr_num['b1'])))
ax.bar(1, np.mean(corr_num['b2']), yerr=np.std(corr_num['b2'])/np.sqrt(len(corr_num['b2'])))
ax.bar(2, np.mean(corr_num['ours']), yerr=np.std(corr_num['ours'])/np.sqrt(len(corr_num['ours'])))

ax.set_title('# of Correction Timesteps')
ax.set_xticks(np.arange(len(algo)))
ax.set_xticklabels(algo)

plt.show()
# plt.savefig('data/corr.svg')
# plt.savefig('data/corr.png')

"""PLOT REGRET"""

users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_8', 'user_9', 'user_10', 'user_11', 'user_12']
REGRET = {'b1': [], 'b2': [], 'ours': []}
for idx, user in enumerate(users):
    tasks = ['task_1', 'task_2', 'task_3']
    for idt, task in enumerate(tasks):
        algo = ['b1', 'b2', 'ours']
        for ida, alg in enumerate(algo):
            data = pickle.load(open('data/{}/{}/{}/data.pkl'.format(user, task, alg), 'rb'))
            xi_r = data['traj_learned'][-1]
            xi_star = data['traj_star'][-1]
            theta_star = data['theta_star'][-1]
            reward_r = reward(xi_r, theta_star, idt)
            reward_star = reward(xi_star, theta_star, idt)
            regret = reward_star - reward_r + 1.1
            REGRET[alg].append(regret)

# fig, ax = plt.subplots()
# ax.bar(0, np.mean(REGRET['b1']), yerr=np.std(REGRET['b1'])/np.sqrt(len(REGRET['b1'])))
# ax.bar(1, np.mean(REGRET['b2']), yerr=np.std(REGRET['b2'])/np.sqrt(len(REGRET['b2'])))
# ax.bar(2, np.mean(REGRET['ours']), yerr=np.std(REGRET['ours'])/np.sqrt(len(REGRET['ours'])))


# ax.set_title('Regret')
# ax.set_xticks(np.arange(len(algo)))
# ax.set_xticklabels(algo)

# # plt.show()
# plt.savefig('data/regret.svg')
# plt.savefig('data/regret.png')
# print(REGRET)
print(np.min([np.concatenate((REGRET['b1'], REGRET['b2'], REGRET['ours']))]))


"""PLOT SCATTER CORRECTIONS VS REGRET"""
fig1, ax1 = plt.subplots()

ax1.scatter(REGRET['b1'][:12], corr_num['b1'][:12], color=np.array([186, 186, 186])/255, alpha=0.75, s=100)
ax1.scatter(REGRET['b2'][:12], corr_num['b2'][:12], color=np.array([189, 167, 255])/255, alpha=0.75, s=100)
ax1.scatter(REGRET['ours'][:12], corr_num['ours'][:12], color=np.array([255, 153, 0])/255, alpha=0.75, s=100)
ax1.set_title('TASK 1')
ax1.legend(['b1', 'b2', 'ours'])
# plt.savefig('data/scatter_1.svg')
# plt.savefig('data/scatter_1.png')

fig2, ax2 = plt.subplots()

ax2.scatter(REGRET['b1'][12:24], corr_num['b1'][12:24], color=np.array([186, 186, 186])/255, alpha=0.75, s=100)
ax2.scatter(REGRET['b2'][12:24], corr_num['b2'][12:24], color=np.array([189, 167, 255])/255, alpha=0.75, s=100)
ax2.scatter(REGRET['ours'][12:24], corr_num['ours'][12:24], color=np.array([255, 153, 0])/255, alpha=0.75, s=100)
ax2.set_title("TASK 2")
ax2.legend(['b1', 'b2', 'ours'])
# plt.savefig('data/scatter_2.svg')
# plt.savefig('data/scatter_2.png')

fig3, ax3 = plt.subplots()

ax3.scatter(REGRET['b1'][24:], corr_num['b1'][24:], color=np.array([186, 186, 186])/255, alpha=0.75, s=100)
ax3.scatter(REGRET['b2'][24:], corr_num['b2'][24:], color=np.array([189, 167, 255])/255, alpha=0.75, s=100)
ax3.scatter(REGRET['ours'][24:], corr_num['ours'][24:], color=np.array([255, 153, 0])/255, alpha=0.75, s=00)
ax3.set_title("TASK 3")
ax3.legend(['b1', 'b2', 'ours'])
# plt.savefig('data/scatter_3.svg')
# plt.savefig('data/scatter_3.png')



fig4, ax4 = plt.subplots()

ax4.scatter(REGRET['b1'], corr_num['b1'], color=np.array([186, 186, 186])/255, alpha=0.75, s=200)
ax4.scatter(REGRET['b2'], corr_num['b2'], color=np.array([189, 167, 255])/255, alpha=0.75, s=200)
ax4.scatter(REGRET['ours'], corr_num['ours'], color=np.array([255, 153, 0])/255, alpha=0.75, s=200)
ax4.set_title("TASK ALL")
ax4.legend(['b1', 'b2', 'ours'])
# plt.savefig('data/scatter_all.svg')
# plt.savefig('data/scatter_all.png')

# plt.show()


"""PLOT LIKERT"""

q1_means = np.array([3.597222222, 5.013888889, 6.430555556])
q1_std = np.array([2.307811098, 2.044106899, 0.903586329])/np.sqrt(36)

q2_means = np.array([2.930555556, 4.388888889, 5.652777778])
q2_std = np.array([1.785468237, 2.003964325, 1.361823591])/np.sqrt(36)

fig, ax = plt.subplots(1,2)
ax[0].set_xticks(np.arange(len(algo)))
ax[0].set_xticklabels(algo)
ax[0].bar(np.arange(len(algo)), q1_means, yerr=q1_std)
ax[0].set_ylim(0, 7)
ax[0].set_title('Learned')
ax[1].set_xticks(np.arange(len(algo)))
ax[1].set_xticklabels(algo)
ax[1].bar(np.arange(len(algo)), q2_means, yerr=q2_std)
ax[1].set_ylim(0, 7)
ax[1].set_title('Intuitive')
# plt.savefig('data/likert.svg')
# plt.savefig('data/likert.png')
# plt.show()