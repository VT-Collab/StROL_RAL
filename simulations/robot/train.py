import numpy as np
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm
from model import Model, Model_end2end, g_function_eval


class TRAIN():
	def __init__(self, args):
		self.args = args
		self.train(args)

	def inv_g(self, x, U, theta0, theta_star, objs, g, gamma):
		n_samples = len(U)
		theta_err = np.array([0.]*n_samples)
		for idx, u_h in enumerate(U):
			theta_estimate = theta0 + gamma*g(x, u_h, objs, self.args)
			theta_err[idx] = torch.linalg.norm(theta_star - theta_estimate).detach().numpy()
		opt_idx = np.argmin(theta_err)
		return U[opt_idx]

	def train(self, args):
		print("GENERATING DATASET")

		gamma = 1.0
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		
		input_size = env_dim + env_dim
		output_size = n_features
		hidden_size = args.hidden_size
		LR = args.lr

		# model = Model_end2end(input_size, output_size, hidden_size)
		model = Model(input_size, output_size, hidden_size)
		
		optimizer = optim.Adam(model.parameters(), lr=LR)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.lr_gamma)

		laptop = np.array([0.35, 0.1, 0.0])
		table = np.array([0.0, 0.0, 0.0])
		goal = np.array([0.7, -0.45, 0.1])
		
		theta_s = np.zeros((n_features, n_features))
		for obj_idy in range(n_features):
			for obj_idx in range(n_features):
				theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/n_features)):
			THETA[idx*n_features:idx*n_features+n_features] = theta_s + np.random.normal(0, 0.1, (n_features, n_features))

		# theta_s = np.zeros((2*n_features, n_features))
		# for obj_idy in range(n_features):
		# 	for obj_idx in range(n_features):
		# 		theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		# for obj_idy in range(n_features):
		# 	for obj_idx in range(n_features):
		# 		theta_s[2+obj_idx, obj_idy] = (-1)**(obj_idx) * 0.8

		# THETA = np.zeros((n_samples, n_features))
		# for idx in range(int(n_samples/n_features/2)):
		# 	THETA[idx*2*n_features:idx*2*n_features+2*n_features] = theta_s + np.random.normal(0, 0.1, (2*n_features, n_features))

		n_batch = args.batch_size
		EPOCH = args.n_train_steps

		for epoch in tqdm(range(EPOCH)):
			u_opt_arr = torch.zeros((n_batch, env_dim))
			pos_arr = torch.zeros((n_batch, env_dim))
			theta_star_arr = torch.zeros((n_batch, n_features))
			theta0_arr = torch.zeros((n_batch, n_features))
			obj_arr = torch.zeros(n_batch, n_features*env_dim)

			for idx in range (n_batch):
				x = 0.2 + 0.5*np.random.rand()
				y = -0.4 + 0.9*np.random.rand()
				z = 0.3 + 0.3*np.random.rand()
				pos = torch.FloatTensor(np.array([x, y, z]))

				table = np.array([x, y, 0.0])
				objs = np.concatenate((laptop, goal))
				goals = torch.FloatTensor(objs)

				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

				theta0 = torch.FloatTensor(1 - 2*np.random.rand(n_features))
				U = 0.1 - 0.2*np.random.rand(100, env_dim)
				U = torch.FloatTensor(np.concatenate((U, np.zeros((1, env_dim)))))

				u_opt = self.inv_g(pos, U, theta0, theta_star, goals, g_function_eval, gamma)

				u_opt_arr[idx, :] = u_opt
				pos_arr[idx, :] = pos
				theta0_arr[idx, :] = theta0
				theta_star_arr[idx, :] = theta_star
				obj_arr[idx, :] = goals
			bias = torch.FloatTensor([0.0, 0.0, 0.0])
			deltas = torch.randn_like(u_opt_arr)*0.025 + bias

			err_theta = theta_star_arr - theta0_arr
			g_tilde = model.g_tilde(pos_arr, u_opt_arr + deltas, obj_arr, args)
			loss1 = torch.sum(torch.norm(g_tilde, dim=1)**2)
			loss2 = torch.sum(err_theta * g_tilde)

			loss = gamma*loss1 - 2*loss2

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			tqdm.write("loss at epoch {} is {}".format(epoch, loss))
			if epoch%100 == 0:
				torch.save(model.state_dict(), 'g_data/model_' + str(args.n_features) + 'objs_' + str(epoch))




"""
For simulations, there should be at least 2 features (say height from table and dist from laptop) that the robot needs to learn. 
We treat the table and the laptop as two objects in the environment
The robot always knows the goal/moves towards the goal. 
The human intends to teach the robot and always knonws the robot's theta.
The human stops teaching once the robot's theta reaches close to its own.

workspace : 
X -> [0.2, 0.7], Y -> [-0.4, 0.5], Z -> [0.0, 0.6]
Table -> [X, Y, 0.0]
Laptop -> [0.35, 0.1, 0.0]
"""


"""
The robot is executing a trajectory. 
When the human feels that the robot is deviating from their intended trajectory, the human jumps in and starts providing a correction (say for 5 timesteps).
The robot looks at these 5 timesteps and tries to learn the human's intended reward weights theta_star.
"""

"""
OUTLINE FOR SIMULATION 1 (CORRECTIONS):
yES
1. The robot is preforming a trajectory towards a goal from a fixed starting point. (use traj_opt for this trajectory or just use a perfect model for taking actions towards the goal from the fixed start point).
2. Track the error from the human's intended actions. if error > threshold, the human comes in to provide a correction
3. The human gives the robot 5 continuous inputs. 
4. Update the reward weights theta based on this input and replan the motion/future actions of the robot.
	a. For ours, we just use the g functions and the human's actions to learn theta
	b. For one feature at a time, we use the actions to see which feature changes most according to g and update the theta for that feature
	c. for misspecified objective functions, we check if the human inputs are relevant to any of the features in the environment and update the theta according to g if they are relevant

NOTE: The robot updates theta based on each input the human provides, i.e. for 5 inputs, the weights are updated 5 times.
"""


"""
Try the evaluation with boltzmann human as well
"""