import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys
from model import Model, g_function_eval, Model_end2end
from trajopt import TrajOpt

def draw_interaction(ax, xi, col='b', alpha=1.0):
	xi_prime = xi
	color = col + 'o-'
	ax.plot(xi_prime[0, 0], xi_prime[0, 1], xi_prime[0, 2], 'gs', markersize=10)
	ax.plot(xi_prime[-1, 0], xi_prime[-1, 1], xi_prime[-1, 2], 'rs', markersize=10)
	ax.plot(xi_prime[:, 0], xi_prime[:, 1], xi_prime[:, 2], color, alpha=alpha)


class EVALUATE():
	def __init__(self, args):
		self.args = args

		if not os.path.exists('results/test/noise_{}_bias_{}'.format(args.noise, args.bias)):
			os.makedirs('results/test/noise_{}_bias_{}'.format(args.noise, args.bias))
			
		self.save_name = 'results/test/noise_{}_bias_{}'.format(args.noise, args.bias)

		self.eval_ours()
		self.eval_end2end()
		self.eval_b1()
		self.eval_b2()
		self.eval_b3()
		self.regret_plot()

	# inverse g model for providing corrections
	def inv_g(self, x, U, theta0, theta_star, objs, g, gamma):
		n_samples = len(U)
		theta_err = np.array([0.]*n_samples)
		for idx, u_h in enumerate(U):
			theta_estimate = theta0 + gamma*g(x, u_h, objs, self.args)
			theta_err[idx] = torch.linalg.norm(theta_star - theta_estimate).detach().numpy()
		opt_idx = np.argmin(theta_err)
		return U[opt_idx]

	# cost function for the task
	def cost(self, x, u, objs, theta):
		cost = 0.
		for idx, object in enumerate(objs):
			if idx == 0:
				cost += theta[idx] * (np.linalg.norm(x[:2] + u[:2] - object[:2]) - np.linalg.norm(x[:2] - object[:2]))
			elif idx == 1:
				cost += theta[idx] * (np.linalg.norm(x + u - object) - np.linalg.norm(x - object))
		return cost

	# boltzmann human for providing correction
	def boltzmann_model(self, x, U, objs, theta, beta):
		P = np.array([0.] * len(U))
		for idx, u in enumerate(U):
			cost = self.cost(x, u, objs, theta)
			P[idx] = np.exp(-beta * cost)
		P /= np.sum(P)
		idx_star = np.argmax(P)
		return U[idx_star]

	# constraint for belief update
	def constraint(self, x, u, objs, theta, theta_star, g, gamma, args):
		e_theta = torch.FloatTensor(theta_star) - theta
		g = g(x, u, objs, args)
		term1 = torch.linalg.norm(g)**2
		term2 = torch.sum(e_theta * g)
		return gamma * term1 - 2 * term2

	# belief update over theta for point estimate learning rule
	def update_belief(self, b, x, u, objs, theta, THETA, gamma, g, args, beta=1.0):
		P = np.zeros(len(b))
		for idx, theta_star in enumerate(THETA):
			cost = self.constraint(x, u, objs, theta, theta_star, g, gamma, args)
			P[idx] = np.exp(-beta * cost.detach().numpy())
		b *= P
		return b/np.sum(b)

	def get_reward(self, xi, theta):
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


	# EVALUATE OUR APPROACH
	def eval_ours(self):
		print("[*] EVALUATING OUR APPROACH")
		save_data = {'theta_star': [], 'theta_learned': [], 'traj_ideal': [], 'traj_learned': [], 'traj_c': [], 'traj_original': []}
		# initialize the environment parameters
		args = self.args
		n_eval = args.n_eval
		n_features = args.n_features
		env_dim = args.env_dim
		n_samples = 3000
		gamma = 1.0

		# initialize model parameters and load the learned g_hat model
		input_size = env_dim + env_dim
		output_size = n_features
		hidden_size = args.hidden_size

		model = Model(input_size, output_size, hidden_size)
		model.load_state_dict(torch.load('g_data/g_tilde/model_' + str(n_features) + 'objs_500'))

		# define the features in the environment and the start pose of the robot 
		laptop = np.array([0.35, 0.1, 0.0])
		goal = np.array([0.7, -0.45, 0.1]) 

		features = torch.FloatTensor(np.array([laptop, goal]))


		theta_s = np.zeros((n_features, n_features))
		for obj_idy in range(n_features):
			for obj_idx in range(n_features):
				theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/n_features)):
			THETA[idx*n_features:idx*n_features+n_features] = theta_s + np.random.normal(0, 0.1, (n_features, n_features))

		# evaluate for n_eval runs
		for _ in tqdm(range (args.n_eval)):

			start_pos = np.array([0.2, 0.2, 0.6]) + np.random.randn(env_dim)*0.0

			# define the initial belief of the robot and the default trajectory to the goal
			theta_0 = np.append(np.array([0.]*2)/2, 0.5)

			# define the human's theta_star
			if args.uniform:
				theta_star = torch.FloatTensor(1- 2*np.random.rand(n_features))
			else:
				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

			# define the action set in the environment
			U = 0.1 - 0.2*np.random.rand(500, env_dim)
			U = torch.FloatTensor(np.concatenate((U, np.zeros((1, env_dim)))))

			# check if the actions taken by the robot are aligned with the intended human actions
			x = torch.FloatTensor(np.copy(start_pos))
			theta = torch.FloatTensor(np.copy(theta_0[:2]))

			trajopt = TrajOpt(args, start_pos, goal, env_dim, waypoints=15)
			traj_ideal, _, _ = trajopt.optimize(theta_star, goal)

			# define the correction parameters and the bias in human actions
			n_correction_steps = 5
			bias = torch.FloatTensor(np.array([args.bias, args.bias, args.bias]))

			theta_ours = torch.clone(theta)
			pos_ours = torch.clone(x)
			xi_c = []
			# provide a correction for a fixed number of steps and learn theta from each step
			for _ in range(n_correction_steps):
				xi_c.append(torch.clone(pos_ours).detach().numpy())
				if args.inv_g:
					u_opt = self.inv_g(pos_ours, U, theta_ours, torch.clone(theta_star), features[:2], g_function_eval, gamma)
				if args.boltzmann:
					u_opt = self.boltzmann_model(pos_ours, U, features[:2], torch.clone(theta_star), beta=5.)
				deltas = torch.randn_like(u_opt)*args.noise + bias
				theta_ours += gamma * model.g_tilde_eval(pos_ours, u_opt+deltas, features[:2], args)[-1]
				pos_ours += u_opt + deltas

			# for _ in range(2):
			# 	xi_c.append(torch.clone(pos_ours).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_ours, U, theta_ours, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_ours, U, features[:2], torch.clone(theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	theta_ours += gamma * model.g_tilde_eval(pos_ours, u_opt+deltas, features[:2], args)[-1]
			# 	pos_ours += u_opt + deltas

			# for _ in range(3):
			# 	xi_c.append(torch.clone(pos_ours).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_ours, U, theta_ours, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_ours, U, features[:2], torch.clone(-1*theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	theta_ours += gamma * model.g_tilde_eval(pos_ours, u_opt+deltas, features[:2], args)[-1]
			# 	pos_ours += u_opt + deltas
			# xi_c.append(torch.clone(pos_ours).detach().numpy())
			# init_traj[1] = torch.clone(pos_ours).detach().numpy()

			# plan and plot trajectory according to the new theta learned
			trajopt = TrajOpt(args, pos_ours.detach().numpy(), goal, env_dim, waypoints=10)
			
			xi_ours, _, _ = trajopt.optimize(theta_ours.detach().numpy(), goal)
			xi_ideal, _, _ = trajopt.optimize(1*theta_star.detach().numpy(), goal)

			save_data['theta_star'].append(torch.clone(theta_star).detach().numpy())
			save_data['theta_learned'].append(theta_ours.detach().numpy())
			save_data['traj_ideal'].append(xi_ideal)
			save_data['traj_learned'].append(xi_ours)
			save_data['traj_c'].append(np.array(xi_c))
			save_data['traj_original'].append(traj_ideal)
		pickle.dump(save_data, open(self.save_name + '/data_ours_lambda_1.0.pkl', 'wb'))


	# EVALUATE END TO END TRAINED MODEL FOR G_HAT
	def eval_end2end(self):
		print("[*] EVALUATING END-2-END APPROACH")
		save_data = {'theta_star': [], 'theta_learned': [], 'traj_ideal': [], 'traj_learned': [], 'traj_c': [], 'traj_original': []}
		# initialize the environment parameters
		args = self.args
		n_eval = args.n_eval
		n_features = args.n_features
		env_dim = args.env_dim
		n_samples = 3000
		waypoints = 5
		gamma = 1.0

		# initialize model parameters and load the learned g_hat model
		input_size = env_dim + env_dim
		output_size = n_features
		hidden_size = args.hidden_size

		model = Model_end2end(input_size, output_size, hidden_size)
		model.load_state_dict(torch.load('g_data/model_' + str(n_features) + 'objs_500'))

		# define the features in the environment and the start pose of the robot 
		laptop = np.array([0.35, 0.1, 0.0])
		goal = np.array([0.7, -0.45, 0.1]) 

		features = torch.FloatTensor(np.array([laptop, goal]))


		theta_s = np.zeros((n_features, n_features))
		for obj_idy in range(n_features):
			for obj_idx in range(n_features):
				theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/n_features)):
			THETA[idx*n_features:idx*n_features+n_features] = theta_s + np.random.normal(0, 0.1, (n_features, n_features))

		# evaluate for n_eval runs
		for _ in tqdm(range (args.n_eval)):

			start_pos = np.array([0.2, 0.2, 0.6]) + np.random.randn(env_dim)*0.0

			# define the initial belief of the robot and the default trajectory to the goal
			theta_0 = np.append(np.array([0.]*2)/2, 0.5)

			# define the human's theta_star
			if args.uniform:
				theta_star = torch.FloatTensor(1- 2*np.random.rand(n_features))
			else:
				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

			# define the action set in the environment
			U = U = 0.1 - 0.2*np.random.rand(500, env_dim)
			U = torch.FloatTensor(np.concatenate((U, np.zeros((1, env_dim)))))

			# check if the actions taken by the robot are aligned with the intended human actions
			x = torch.FloatTensor(np.copy(start_pos))
			theta = torch.FloatTensor(np.copy(theta_0[:2]))

			trajopt = TrajOpt(args, start_pos, goal, env_dim, waypoints=15)
			traj_ideal, _, _ = trajopt.optimize(theta_star, goal)
			
			# define the correction parameters and the bias in human actions
			n_correction_steps = 5
			bias = torch.FloatTensor(np.array([args.bias, args.bias, args.bias]))			

			theta_e2e = torch.clone(theta)
			pos_e2e = torch.clone(x)
			xi_c = []
			# provide a correction for a fixed number of steps and learn theta from each step
			for _ in range(n_correction_steps):
				xi_c.append(torch.clone(pos_e2e).detach().numpy())
				if args.inv_g:
					u_opt = self.inv_g(pos_e2e, U, theta_e2e, torch.clone(theta_star), features[:2], g_function_eval, gamma)
				if args.boltzmann:
					u_opt = self.boltzmann_model(pos_e2e, U, features[:2], torch.clone(theta_star), beta=5.)
				deltas = torch.randn_like(u_opt)*args.noise + bias
				theta_e2e += gamma * model.g_tilde_eval(pos_e2e, u_opt+deltas, features[:2], args)[-1]
				pos_e2e += u_opt + deltas

			# for _ in range(2):
			# 	xi_c.append(torch.clone(pos_e2e).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_e2e, U, theta_e2e, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_e2e, U, features[:2], torch.clone(theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	theta_e2e += gamma * model.g_tilde_eval(pos_e2e, u_opt+deltas, features[:2], args)[-1]
			# 	pos_e2e += u_opt + deltas

			# for _ in range(3):
			# 	xi_c.append(torch.clone(pos_e2e).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_e2e, U, theta_e2e, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_e2e, U, features[:2], torch.clone(-1*theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	theta_e2e += gamma * model.g_tilde_eval(pos_e2e, u_opt+deltas, features[:2], args)[-1]
			# 	pos_e2e += u_opt + deltas
			# xi_c.append(torch.clone(pos_e2e).detach().numpy())

			# plan and plot trajectory according to the new theta learned
			trajopt = TrajOpt(args, pos_e2e.detach().numpy(), goal, env_dim, waypoints=10)
			
			xi_e2e, _, _ = trajopt.optimize(theta_e2e.detach().numpy(), goal)
			xi_ideal, _, _ = trajopt.optimize(1*theta_star.detach().numpy(), goal)


			save_data['theta_star'].append(torch.clone(theta_star).detach().numpy())
			save_data['theta_learned'].append(theta_e2e.detach().numpy())
			save_data['traj_ideal'].append(xi_ideal)
			save_data['traj_learned'].append(xi_e2e)
			save_data['traj_original'].append(traj_ideal)
			save_data['traj_c'].append(np.array(xi_c))
		pickle.dump(save_data, open(self.save_name + '/data_e2e.pkl', 'wb'))


	# EVALUATE BASELINE 1 -- ONE FEATURE AT A TIME
	def eval_b1(self):
		print("[*] EVALUATING BASELINE 1")
		save_data = {'theta_star': [], 'theta_learned': [], 'traj_ideal': [], 'traj_learned': [], 'traj_c': [], 'traj_original': []}
		# initialize the environment parameters
		args = self.args
		n_eval = args.n_eval
		n_features = args.n_features
		env_dim = args.env_dim
		n_samples = 3000
		waypoints = 5
		gamma = 1.0

		# define the features in the environment and the start pose of the robot
		laptop = np.array([0.35, 0.1, 0.0])
		goal = np.array([0.7, -0.45, 0.1]) 
		features = torch.FloatTensor(np.array([laptop, goal]))


		theta_s = np.zeros((n_features, n_features))
		for obj_idy in range(n_features):
			for obj_idx in range(n_features):
				theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/n_features)):
			THETA[idx*n_features:idx*n_features+n_features] = theta_s + np.random.normal(0, 0.1, (n_features, n_features))

		# evaluate for n_eval runs
		for _ in tqdm(range(args.n_eval)):
			start_pos = np.array([0.2, 0.2, 0.6]) + np.random.randn(env_dim)*0.0

			# define the initial belief of the robot and the default trajectory to the goal
			theta_0 = np.append(np.array([0.]*2)/2, 0.5)

			# define the human's theta_star
			if args.uniform:
				theta_star = torch.FloatTensor(1- 2*np.random.rand(n_features))
			else:
				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

			# define the action set in the environment
			U = U = 0.1 - 0.2*np.random.rand(500, env_dim)
			U = torch.FloatTensor(np.concatenate((U, np.zeros((1, env_dim)))))

			# check if the actions taken by the robot are aligned with the intended human actions
			x = torch.FloatTensor(np.copy(start_pos))
			theta = torch.FloatTensor(np.copy(theta_0[:2]))
			
			trajopt = TrajOpt(args, start_pos, goal, env_dim, waypoints=15)
			traj_ideal, _, _ = trajopt.optimize(theta_star, goal)

			# define the correction parameters and the bias in human actions
			n_correction_steps = 5
			bias = torch.FloatTensor(np.array([args.bias, args.bias, args.bias]))

			theta_b1 = torch.clone(theta)
			pos_b1 = torch.clone(x)
			xi_c = []
			# provide a correction for a fixed number of steps and learn theta from each step
			for _ in range(n_correction_steps):
				xi_c.append(torch.clone(pos_b1).detach().numpy())
				if args.inv_g:
					u_opt = self.inv_g(pos_b1, U, theta_b1, torch.clone(theta_star), features[:2], g_function_eval, gamma)
				if args.boltzmann:
					u_opt = self.boltzmann_model(pos_b1, U, features[:2], torch.clone(theta_star), beta=5.)
				deltas = torch.randn_like(u_opt)*args.noise + bias
				g_func = g_function_eval(pos_b1, u_opt+deltas, features[:2], args)[-1]
				update_idx = np.argmax(abs(g_func.detach().numpy()))
				theta_b1[update_idx] += gamma * g_func[update_idx]
				pos_b1 += u_opt + deltas

			# for _ in range(2):
			# 	xi_c.append(torch.clone(pos_b1).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_b1, U, theta_b1, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_b1, U, features[:2], torch.clone(theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	g_func = g_function_eval(pos_b1, u_opt+deltas, features[:2], args)[-1]
			# 	update_idx = np.argmax(abs(g_func.detach().numpy()))
			# 	theta_b1[update_idx] += gamma * g_func[update_idx]
			# 	pos_b1 += u_opt + deltas

			# for _ in range(3):
			# 	xi_c.append(torch.clone(pos_b1).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_b1, U, theta_b1, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_b1, U, features[:2], torch.clone(-1*theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	g_func = g_function_eval(pos_b1, u_opt+deltas, features[:2], args)[-1]
			# 	update_idx = np.argmax(abs(g_func.detach().numpy()))
			# 	theta_b1[update_idx] += gamma * g_func[update_idx]
			# 	pos_b1 += u_opt + deltas
			# xi_c.append(torch.clone(pos_b1).detach().numpy())
			
			# plan and plot trajectory according to the new theta learned
			trajopt = TrajOpt(args, pos_b1.detach().numpy(), goal, env_dim, waypoints=10)
			
			xi_b1, _, _ = trajopt.optimize(theta_b1.detach().numpy(), goal)
			xi_ideal, _, _ = trajopt.optimize(1*theta_star.detach().numpy(), goal)

			save_data['theta_star'].append(torch.clone(theta_star).detach().numpy())
			save_data['theta_learned'].append(theta_b1.detach().numpy())
			save_data['traj_ideal'].append(xi_ideal)
			save_data['traj_learned'].append(xi_b1)
			save_data['traj_c'].append(np.array(xi_c))
			save_data['traj_original'].append(traj_ideal)
		pickle.dump(save_data, open(self.save_name + '/data_b1.pkl', 'wb'))


	# EVALUATE BASELINE 2 -- MIS SPECIFIED OBJECTIVE SPACES
	def eval_b2(self):
		print("[*] EVALUATING BASELINE 2")
		save_data = {'theta_star': [], 'theta_learned': [], 'traj_ideal': [], 'traj_learned': [], 'traj_c': [], 'traj_original': []}
		# initialize the environment parameters
		args = self.args
		n_eval = args.n_eval
		n_features = args.n_features
		env_dim = args.env_dim
		n_samples = 3000
		waypoints=5
		gamma = 1.0
		dist_threshold = 1.5

		# define the features in the environment and the start pose of the robot
		laptop = np.array([0.35, 0.1, 0.0])
		goal = np.array([0.7, -0.45, 0.1])

		features = torch.FloatTensor(np.array([laptop, goal]))


		theta_s = np.zeros((n_features, n_features))
		for obj_idy in range(n_features):
			for obj_idx in range(n_features):
				theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/n_features)):
			THETA[idx*n_features:idx*n_features+n_features] = theta_s + np.random.normal(0, 0.1, (n_features, n_features))

		# evaluate for n_eval runs
		for _ in tqdm(range(args.n_eval)):
		
			start_pos = np.array([0.2, 0.2, 0.6]) + np.random.randn(env_dim)*0.0

			# define the initial belief of the robot and the default trajectory to the goal
			theta_0 = np.append(np.array([0.]*2)/2, 0.5)

			# define the human's theta_star
			if args.uniform:
				theta_star = torch.FloatTensor(1- 2*np.random.rand(n_features))
			else:
				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

			# define the action set in the environment
			U = U = 0.1 - 0.2*np.random.rand(500, env_dim)
			U = torch.FloatTensor(np.concatenate((U, np.zeros((1, env_dim)))))

			# check if the actions taken by the robot are aligned with the intended human actions
			x = torch.FloatTensor(np.copy(start_pos))
			theta = torch.FloatTensor(np.copy(theta_0[:2]))

			trajopt = TrajOpt(args, start_pos, goal, env_dim, waypoints=15)
			traj_ideal, _, _ = trajopt.optimize(theta_star, goal)
			
			# define the correction parameters and the bias in human actions
			n_correction_steps = 5
			bias = torch.FloatTensor(np.array([args.bias, args.bias, args.bias]))

			theta_b2 = torch.clone(theta)
			pos_b2 = torch.clone(x)
			xi_c = []
			# provide a correction for a fixed number of steps and learn theta from each step
			for _ in range(n_correction_steps):
				xi_c.append(torch.clone(pos_b2 ).detach().numpy())
				if args.inv_g:
					u_opt = self.inv_g(pos_b2, U, theta_b2, torch.clone(theta_star), features[:2], g_function_eval, gamma)
				if args.boltzmann:
					u_opt = self.boltzmann_model(pos_b2, U, features[:2], torch.clone(theta_star), beta=5.)
				deltas = torch.randn_like(u_opt)*args.noise + bias

				u_h = u_opt + deltas
				theta_star_h = torch.eye(2)

				u_dist = []
				for id_x in range (n_features):
					for id_y in range(n_features):
						theta_star_h *= (-1)**id_x
						if args.inv_g:
							u_opt = self.inv_g(pos_b2, U, theta_b2, torch.clone(theta_star), features[:2], g_function_eval, gamma)
						if args.boltzmann:
							u_opt = self.boltzmann_model(pos_b2, U, features[:2], theta_star_h[id_y], 5.)
						u_dist.append(torch.norm(u_opt - u_h).detach().numpy())
						
				if np.array(u_dist).any() < dist_threshold:
					g_func = g_function_eval(pos_b2, u_h, features[:2], args)[-1]
					theta_b2 += gamma * g_func
				pos_b2 += u_h

			# for _ in range(2):
			# 	xi_c.append(torch.clone(pos_b2 ).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_b2, U, theta_b2, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_b2, U, features[:2], torch.clone(theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias

			# 	u_h = u_opt + deltas
			# 	theta_star_h = torch.eye(2)

			# 	u_dist = []
			# 	for id_x in range (n_features):
			# 		for id_y in range(n_features):
			# 			theta_star_h *= (-1)**id_x
			# 			if args.inv_g:
			# 				u_opt = self.inv_g(pos_b2, U, theta_b2, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 			if args.boltzmann:
			# 				u_opt = self.boltzmann_model(pos_b2, U, features[:2], theta_star_h[id_y], 5.)
			# 			u_dist.append(torch.norm(u_opt - u_h).detach().numpy())
						
			# 	if np.array(u_dist).any() < dist_threshold:
			# 		g_func = g_function_eval(pos_b2, u_h, features[:2], args)[-1]
			# 		theta_b2 += gamma * g_func
			# 	pos_b2 += u_h


			# for _ in range(3):
			# 	xi_c.append(torch.clone(pos_b2 ).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_b2, U, theta_b2, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_b2, U, features[:2], torch.clone(-1*theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias

			# 	u_h = u_opt + deltas
			# 	theta_star_h = torch.eye(2)

			# 	u_dist = []
			# 	for id_x in range (n_features):
			# 		for id_y in range(n_features):
			# 			theta_star_h *= (-1)**id_x
			# 			if args.inv_g:
			# 				u_opt = self.inv_g(pos_b2, U, theta_b2, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 			if args.boltzmann:
			# 				u_opt = self.boltzmann_model(pos_b2, U, features[:2], theta_star_h[id_y], 5.)
			# 			u_dist.append(torch.norm(u_opt - u_h).detach().numpy())
						
			# 	if np.array(u_dist).any() < dist_threshold:
			# 		g_func = g_function_eval(pos_b2, u_h, features[:2], args)[-1]
			# 		theta_b2 += gamma * g_func
			# 	pos_b2 += u_h
			# xi_c.append(torch.clone(pos_b2).detach().numpy())
			
			# plan and plot trajectory according to the new theta learned
			trajopt = TrajOpt(args, pos_b2.detach().numpy(), goal, env_dim, waypoints=10)
			
			xi_b2, _, _ = trajopt.optimize(theta_b2.detach().numpy(), goal)
			xi_ideal, _, _ = trajopt.optimize(1*theta_star.detach().numpy(), goal)


			save_data['theta_star'].append(torch.clone(theta_star).detach().numpy())
			save_data['theta_learned'].append(theta_b2.detach().numpy())
			save_data['traj_ideal'].append(xi_ideal)
			save_data['traj_learned'].append(xi_b2)
			save_data['traj_c'].append(np.array(xi_c))
			save_data['traj_original'].append(traj_ideal)
		pickle.dump(save_data, open(self.save_name + '/data_b2.pkl', 'wb'))
		
	
	# EVALUATE BASELINE 3 -- ALL FEATURES AT A TIME
	def eval_b3(self):
		print("[*] EVALUATING BASELINE 3")
		args = self.args
		save_data = {'theta_star': [], 'theta_learned': [], 'traj_ideal': [], 'traj_learned': [], 'traj_c': [], 'traj_original': []}
		# initialize the environment parameters
		args = self.args
		n_eval = args.n_eval
		n_features = args.n_features
		env_dim = args.env_dim
		n_samples = 3000
		waypoints = 5
		gamma = 1.0

		# define the features in the environment and the start pose of the robot
		laptop = np.array([0.35, 0.1, 0.0])
		goal = np.array([0.7, -0.45, 0.1])

		features = torch.FloatTensor(np.array([laptop, goal]))


		theta_s = np.zeros((n_features, n_features))
		for obj_idy in range(n_features):
			for obj_idx in range(n_features):
				theta_s[obj_idx, obj_idy] = (-1)**(obj_idx+obj_idy) * 0.8

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/n_features)):
			THETA[idx*n_features:idx*n_features+n_features] = theta_s + np.random.normal(0, 0.1, (n_features, n_features))

		# evaluate for n_eval runs
		for _ in tqdm(range(args.n_eval)):
			start_pos = np.array([0.2, 0.2, 0.6]) + np.random.randn(env_dim)*0.0

			# define the initial belief of the robot and the default trajectory to the goal
			theta_0 = np.append(np.array([0.]*2)/2, 0.5)

			# define the human's theta_star
			if args.uniform:
				theta_star = torch.FloatTensor(1- 2*np.random.rand(n_features))
			else:
				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

			# define the action set in the environment
			U = U = 0.1 - 0.2*np.random.rand(500, env_dim)
			U = torch.FloatTensor(np.concatenate((U, np.zeros((1, env_dim)))))

			# check if the actions taken by the robot are aligned with the intended human actions
			x = torch.FloatTensor(np.copy(start_pos))
			theta = torch.FloatTensor(np.copy(theta_0[:2]))

			trajopt = TrajOpt(args, start_pos, goal, env_dim, waypoints=15)
			traj_ideal, _, _ = trajopt.optimize(theta_star, goal)
			
			# define the correction parameters and the bias in human actions
			n_correction_steps = 5
			bias = torch.FloatTensor(np.array([args.bias, args.bias, args.bias]))

			theta_b3 = torch.clone(theta)
			pos_b3 = torch.clone(x)
			xi_c = []
			# provide a correction for a fixed number of steps and learn theta from each step
			for _ in range(n_correction_steps):
				xi_c.append(torch.clone(pos_b3).detach().numpy())
				if args.inv_g:
					u_opt = self.inv_g(pos_b3, U, theta_b3, torch.clone(theta_star), features[:2], g_function_eval, gamma)
				if args.boltzmann:
					u_opt = self.boltzmann_model(pos_b3, U, features[:2], torch.clone(theta_star), beta=5.)
				deltas = torch.randn_like(u_opt)*args.noise + bias
				g_func = g_function_eval(pos_b3, u_opt+deltas, features[:2], args)[-1]
				theta_b3 += gamma * g_func
				pos_b3 += u_opt + deltas

			# for _ in range(2):
			# 	xi_c.append(torch.clone(pos_b3).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_b3, U, theta_b3, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_b3, U, features[:2], torch.clone(theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	g_func = g_function_eval(pos_b3, u_opt+deltas, features[:2], args)[-1]
			# 	theta_b3 += gamma * g_func
			# 	pos_b3 += u_opt + deltas

			# for _ in range(3):
			# 	xi_c.append(torch.clone(pos_b3).detach().numpy())
			# 	if args.inv_g:
			# 		u_opt = self.inv_g(pos_b3, U, theta_b3, torch.clone(theta_star), features[:2], g_function_eval, gamma)
			# 	if args.boltzmann:
			# 		u_opt = self.boltzmann_model(pos_b3, U, features[:2], torch.clone(-1*theta_star), beta=5.)
			# 	deltas = torch.randn_like(u_opt)*args.noise + bias
			# 	g_func = g_function_eval(pos_b3, u_opt+deltas, features[:2], args)[-1]
			# 	theta_b3 += gamma * g_func
			# 	pos_b3 += u_opt + deltas
			# xi_c.append(torch.clone(pos_b3).detach().numpy())
			
			# plan and plot trajectory according to the new theta learned
			trajopt = TrajOpt(args, pos_b3.detach().numpy(), goal, env_dim, waypoints=10)
			
			xi_b3, _, _ = trajopt.optimize(theta_b3.detach().numpy(), goal)
			xi_ideal, _, _ = trajopt.optimize(1*theta_star.detach().numpy(), goal)

			save_data['theta_star'].append(torch.clone(theta_star).detach().numpy())
			save_data['theta_learned'].append(theta_b3.detach().numpy())
			save_data['traj_ideal'].append(xi_ideal)
			save_data['traj_learned'].append(xi_b3)
			save_data['traj_c'].append(np.array(xi_c))
			save_data['traj_original'].append(traj_ideal)
		pickle.dump(save_data, open(self.save_name + '/data_b3.pkl', 'wb'))


	# PLOT ERROR AND REGRET
	def regret_plot(self):
		args = self.args
	
		algo = ['ours', 'e2e', 'b1', 'b2', 'b3']

		fig1, ax1 = plt.subplots()
		fig2, ax2 = plt.subplots()

		for alg_id, alg in enumerate(algo):
			regret = []
			data = pickle.load(open(self.save_name + '/data_' + alg + '.pkl', 'rb'))

			err_theta = np.linalg.norm(np.array(data['theta_star']) - np.array(data['theta_learned']), axis = 1)

			xi = data['traj_learned']
			xi_ideal = data['traj_ideal']
			xi_c = data['traj_c']

			

			for idx in range(len(xi)):

				reward_ideal = self.get_reward(xi_ideal[idx], 1*data['theta_star'][idx])
				reward = self.get_reward(xi[idx], 1*data['theta_star'][idx])

				regret.append(reward_ideal - reward)

			ax1.bar(alg_id, np.mean(regret), yerr=np.std(regret)/np.sqrt(len(regret)))
			
			ax2.bar(alg_id, np.mean(err_theta), yerr=np.std(err_theta)/np.sqrt(len(err_theta)))

			print(alg, np.mean(regret))

		ax1.set_xticks(np.arange(len(algo)))
		ax1.set_xticklabels(algo)
		ax1.set_title('regret (noise = ' + str(args.noise) + ', bias = ' + str(args.bias) + ')')

		ax2.set_xticks(np.arange(len(algo)))
		ax2.set_xticklabels(algo)
		ax2.set_title('theta error (noise = ' + str(args.noise) + ', bias = ' + str(args.bias))

		fig1.savefig(self.save_name + '/regret.svg')
		fig1.savefig(self.save_name + '/regret.png')
		fig2.savefig(self.save_name + '/theta_error.svg')
		fig2.savefig(self.save_name + '/theta_error.png')

		# plt.close()
		# plt.show()
