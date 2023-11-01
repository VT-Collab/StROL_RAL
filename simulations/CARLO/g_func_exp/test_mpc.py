import numpy as np
import pickle
import torch
import time
import os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from g_func_exp.model import Model, Model_end2end, g_function_eval
from mpc_highway import MPC

class EVALUATE():
	def __init__(self, args):
		self.args = args
		if not os.path.exists('g_func_exp/results/noise_{}_bias_{}'.format(args.noise, args.bias)):
			os.makedirs('g_func_exp/results/noise_{}_bias_{}'.format(args.noise, args.bias))

		self.save_name = 'g_func_exp/results/noise_{}_bias_{}'.format(args.noise, args.bias)


		self.eval_ours(args)
		self.eval_e2e(args)
		self.eval_b1(args)
		self.eval_b2(args)
		self.eval_b3(args)
		self.plot(args)

	def inv_g(self, x, U, vel, heading, theta0, theta_star, g, gamma):
		n_samples = len(U)
		theta_err = np.array([0.]*n_samples)
		for idx, u_h in enumerate(U):
			theta_estimate = theta0 + gamma*g(x, u_h, vel, heading, self.args)
			theta_err[idx] = torch.linalg.norm(theta_star - theta_estimate).detach().numpy()
		opt_idx = np.argmin(theta_err)
		return U[opt_idx]

	# EVALUATE OUR APPROACH
	def eval_ours(self, args):
		print("[*] EVALUATING OUR APPROACH")
		save_data  = {'xi': [], 'u': [], 'vel': [], 'theta_star': [], 'theta_predicted': [], 'cost': []}

		n_eval = args.n_eval
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		gamma = 1.0

		input_size = 7
		output_size = n_features
		hidden_size = args.hidden_size
		LR = args.lr

		model = Model(input_size, output_size, hidden_size)
		model.load_state_dict(torch.load('g_func_exp/g_data/Ours/hidden' + str(args.hidden_size) + '/model_' + str(n_features) + 'objs_900'))
		
		theta_s = np.array([[1.0, 1.0, 0.1], [0.1, -1.0, 1.0]])

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/len(theta_s))):
			THETA[idx*len(theta_s):idx*len(theta_s)+len(theta_s)] = theta_s + np.random.normal(0, 0.1, (len(theta_s), n_features))

		U = np.transpose(np.vstack((np.array([0.]*500), np.array([0.]*500), 0.5 - 1*np.random.rand(500), 0.5 - 1*np.random.rand(500))))

		U = torch.FloatTensor(np.concatenate((U, np.array([[0., 0.0, 0., 0.]]))))
		
		for _ in tqdm(range(n_eval)):
			mpc = MPC(3, 10)
			w, c1, c2 = mpc.create_world()
			if args.uniform:
				theta_star = 1 - 2*np.random.rand(n_features)
			else:
				theta_star = THETA[np.random.choice(n_samples)]
				theta_star[0] = np.clip(theta_star[0], 0, 1.5)
				theta_star[2] = np.clip(theta_star[2], 0, 1.5)

			theta_0 = np.zeros(n_features)

			theta_ours = np.copy(theta_0)

			n_obs_steps = 9
			u_arr = []
			count = 0
			cost = 0

			xi_arr = []
			u_opt_arr = []
			vel_arr = []
			

			
			for idx in range(10):
				pos = torch.FloatTensor(np.array([c1.center.x, c1.center.y, c2.center.x, c2.center.y]))

				vel = torch.FloatTensor(np.array([c1.velocity.x, c1.velocity.y, c2.velocity.x, c2.velocity.y]))

				heading = torch.FloatTensor(np.array([c1.heading, c2.heading]))

				mpc.update_world(w)
				u_h = mpc.optimize_h(theta_star, args)
				u_r = mpc.optimize_r(theta_ours, count, args)

				bias = np.array([args.bias, args.bias])
				u_h += np.random.rand(2)*args.noise + bias

				u_opt = torch.FloatTensor(np.array([u_r[0], u_r[1], u_h[0], u_h[1]]))

				xi_arr.append(pos.detach().numpy())
				u_opt_arr.append(u_opt.detach().numpy())
				vel_arr.append(vel.detach().numpy())

				u_arr.append([u_r[0], u_r[1], u_h[0], u_h[1]])

				if count <= n_obs_steps:
					u_opt = torch.FloatTensor(u_opt)

					theta_ours += gamma * model.g_tilde_eval(pos, u_opt, vel, heading, args).detach().numpy()
				
				c1.set_control(u_r[0], u_r[1])
				c2.set_control(u_h[0], u_h[1])
				w.tick()
				# w.render()
				# time.sleep(w.dt/4)
				count += 1

			save_data['xi'].append(xi_arr)
			save_data['u'].append(u_opt_arr)
			save_data['vel'].append(vel_arr)
			save_data['theta_star'].append(theta_star)
			save_data['theta_predicted'].append(theta_ours/np.linalg.norm(theta_ours)*np.max(abs(theta_star)))

			w.close()

		pickle.dump(save_data, open(self.save_name + '/ours.pkl', 'wb'))

	# EVALUATE END-2-END APPROACH
	def eval_e2e(self, args):
		print("[*] EVALUATING END-2-END APPROACH")
		save_data  = {'xi': [], 'u': [], 'vel': [], 'theta_star': [], 'theta_predicted': [], 'cost': []}

		n_eval = args.n_eval
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		gamma = 1.0

		input_size = 7
		output_size = n_features
		hidden_size = args.hidden_size
		LR = args.lr

		model = Model_end2end(input_size, output_size, hidden_size)
		model.load_state_dict(torch.load('g_func_exp/g_data/e2e/hidden' + str(args.hidden_size) + '/model_' + str(n_features) + 'objs_500'))

		theta_s = np.array([[1.0, 1.0, 0.1], [0.1, -1.0, 1.0]])

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/len(theta_s))):
			THETA[idx*len(theta_s):idx*len(theta_s)+len(theta_s)] = theta_s + np.random.normal(0, 0.1, (len(theta_s), n_features))

		U = np.transpose(np.vstack((np.array([0.]*500), np.array([0.]*500), 0.5 - 1*np.random.rand(500), 0.5 - 1*np.random.rand(500))))

		U = torch.FloatTensor(np.concatenate((U, np.array([[0., 0.0, 0., 0.]]))))
		
		for _ in tqdm(range(n_eval)):
			mpc = MPC(3, 10)
			w, c1, c2 = mpc.create_world()
			if args.uniform:
				theta_star = 1 - 2*np.random.rand(n_features)
			else:
				theta_star = THETA[np.random.choice(n_samples)]
				theta_star[0] = np.clip(theta_star[0], 0, 1.5)
				theta_star[2] = np.clip(theta_star[2], 0, 1.5)

			theta_0 = np.zeros(n_features)

			theta_e2e = np.copy(theta_0)

			n_obs_steps = 9
			u_arr = []
			count = 0
			cost = 0

			xi_arr = []
			u_opt_arr = []
			vel_arr = []
			

			
			for idx in range(10):
				pos = torch.FloatTensor(np.array([c1.center.x, c1.center.y, c2.center.x, c2.center.y]))

				vel = torch.FloatTensor(np.array([c1.velocity.x, c1.velocity.y, c2.velocity.x, c2.velocity.y]))

				heading = torch.FloatTensor(np.array([c1.heading, c2.heading]))

				mpc.update_world(w)
				u_h = mpc.optimize_h(theta_star, args)
				u_r = mpc.optimize_r(theta_e2e, count, args)

				bias = np.array([args.bias, args.bias])
				u_h += np.random.rand(2)*args.noise + bias

				u_opt = torch.FloatTensor(np.array([u_r[0], u_r[1], u_h[0], u_h[1]]))

				xi_arr.append(pos.detach().numpy())
				u_opt_arr.append(u_opt.detach().numpy())
				vel_arr.append(vel.detach().numpy())

				u_arr.append([u_r[0], u_r[1], u_h[0], u_h[1]])

				if count <= n_obs_steps:
					u_opt = torch.FloatTensor(u_opt)

					theta_e2e += gamma * model.g_tilde_eval(pos, u_opt, vel, heading, args).detach().numpy()
				
				c1.set_control(u_r[0], u_r[1])
				c2.set_control(u_h[0], u_h[1])
				w.tick()
				# w.render()
				# time.sleep(w.dt/4)
				count += 1

			save_data['xi'].append(xi_arr)
			save_data['u'].append(u_opt_arr)
			save_data['vel'].append(vel_arr)
			save_data['theta_star'].append(theta_star)
			save_data['theta_predicted'].append(theta_e2e/np.linalg.norm(theta_e2e)*np.max(abs(theta_star)))

			w.close()

		pickle.dump(save_data, open(self.save_name + '/e2e.pkl', 'wb'))

	# EVALUATE GRADIENT DESCENT
	def eval_b1(self, args):
		print("[*] EVALUATING BASELINE 1")

		save_data  = {'xi': [], 'u': [], 'vel': [], 'theta_star': [], 'theta_predicted': [], 'cost': []}

		n_eval = args.n_eval
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		gamma = 1.0

		theta_s = np.array([[1.0, 1.0, 0.1], [0.1, -1.0, 1.0]])

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/len(theta_s))):
			THETA[idx*len(theta_s):idx*len(theta_s)+len(theta_s)] = theta_s + np.random.normal(0, 0.1, (len(theta_s), n_features))

		U = np.transpose(np.vstack((np.array([0.]*500), np.array([0.]*500), 0.5 - 1*np.random.rand(500), 0.5 - 1*np.random.rand(500))))

		U = torch.FloatTensor(np.concatenate((U, np.array([[0., 0.0, 0., 0.]]))))
		
		for _ in tqdm(range(n_eval)):
			mpc = MPC(3, 10)
			w, c1, c2 = mpc.create_world()
			if args.uniform:
				theta_star = 1 - 2*np.random.rand(n_features)
			else:
				theta_star = THETA[np.random.choice(n_samples)]
				theta_star[0] = np.clip(theta_star[0], 0, 1.5)
				theta_star[2] = np.clip(theta_star[2], 0, 1.5)

			theta_0 = np.zeros(n_features)

			theta_b1 = np.copy(theta_0)

			n_obs_steps = 9
			u_arr = []
			count = 0
			cost = 0

			xi_arr = []
			u_opt_arr = []
			vel_arr = []
			
			for idx in range(10):
				pos = torch.FloatTensor(np.array([c1.center.x, c1.center.y, c2.center.x, c2.center.y]))

				vel = torch.FloatTensor(np.array([c1.velocity.x, c1.velocity.y, c2.velocity.x, c2.velocity.y]))

				heading = torch.FloatTensor(np.array([c1.heading, c2.heading]))

				mpc.update_world(w)
				u_h = mpc.optimize_h(theta_star, args)
				u_r = mpc.optimize_r(theta_b1, count, args)

				bias = np.array([args.bias, args.bias])

				u_h += np.random.rand(2)*args.noise + bias

				u_opt = torch.FloatTensor(np.array([u_r[0], u_r[1], u_h[0], u_h[1]]))

				xi_arr.append(pos.detach().numpy())
				u_opt_arr.append(u_opt.detach().numpy())
				vel_arr.append(vel.detach().numpy())

				u_arr.append([u_r[0], u_r[1], u_h[0], u_h[1]])

				if count <= n_obs_steps:
					u_opt = torch.FloatTensor(u_opt)

					theta_b1 +=  gamma * g_function_eval(pos, u_opt, vel, heading, args).detach().numpy()

				c1.set_control(u_r[0], u_r[1])
				c2.set_control(u_h[0], u_h[1])
				w.tick()
				# w.render()
				# time.sleep(w.dt/4)
				count += 1

			save_data['xi'].append(xi_arr)
			save_data['u'].append(u_opt_arr)
			save_data['vel'].append(vel_arr)
			save_data['theta_star'].append(theta_star)
			save_data['theta_predicted'].append(theta_b1/np.linalg.norm(theta_b1)*np.max(abs(theta_star)))

			w.close()

		pickle.dump(save_data, open(self.save_name + '/b1.pkl', 'wb'))

	# EVALUATE ONE FEATURE AT A TIME
	def eval_b2(self, args):
		print("[*] EVALUATING BASELINE 2")

		save_data  = {'xi': [], 'u': [], 'vel': [], 'theta_star': [], 'theta_predicted': [], 'cost': []}

		n_eval = args.n_eval
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		gamma = 1.0

		theta_s = np.array([[1.0, 1.0, 0.1], [0.1, -1.0, 1.0]])

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/len(theta_s))):
			THETA[idx*len(theta_s):idx*len(theta_s)+len(theta_s)] = theta_s + np.random.normal(0, 0.1, (len(theta_s), n_features))

		U = np.transpose(np.vstack((np.array([0.]*500), np.array([0.]*500), 0.5 - 1*np.random.rand(500), 0.5 - 1*np.random.rand(500))))

		U = torch.FloatTensor(np.concatenate((U, np.array([[0., 0.0, 0., 0.]]))))
		
		for _ in tqdm(range(n_eval)):
			mpc = MPC(3, 10)
			w, c1, c2 = mpc.create_world()
			if args.uniform:
				theta_star = 1 - 2*np.random.rand(n_features)
			else:
				theta_star = THETA[np.random.choice(n_samples)]
				theta_star[0] = np.clip(theta_star[0], 0, 1.5)
				theta_star[2] = np.clip(theta_star[2], 0, 1.5)

			theta_0 = np.zeros(n_features)

			theta_b2 = np.copy(theta_0)

			n_obs_steps = 9
			u_arr = []
			count = 0
			cost = 0

			xi_arr = []
			u_opt_arr = []
			vel_arr = []
			
			for idx in range(10):
				pos = torch.FloatTensor(np.array([c1.center.x, c1.center.y, c2.center.x, c2.center.y]))

				vel = torch.FloatTensor(np.array([c1.velocity.x, c1.velocity.y, c2.velocity.x, c2.velocity.y]))

				heading = torch.FloatTensor(np.array([c1.heading, c2.heading]))

				mpc.update_world(w)
				u_h = mpc.optimize_h(theta_star, args)
				u_r = mpc.optimize_r(theta_b2, count, args)

				bias = np.array([args.bias, args.bias])

				u_h += np.random.rand(2)*args.noise + bias

				u_opt = torch.FloatTensor(np.array([u_r[0], u_r[1], u_h[0], u_h[1]]))

				xi_arr.append(pos.detach().numpy())
				u_opt_arr.append(u_opt.detach().numpy())
				vel_arr.append(vel.detach().numpy())

				u_arr.append([u_r[0], u_r[1], u_h[0], u_h[1]])

				if count <= n_obs_steps:
					u_opt = torch.FloatTensor(u_opt)

					g =  gamma * g_function_eval(pos, u_opt, vel, heading, args).detach().numpy()

					update_idx = np.argmax(abs(g))
					theta_b2[update_idx] += g[update_idx]

				c1.set_control(u_r[0], u_r[1])
				c2.set_control(u_h[0], u_h[1])
				w.tick()
				# w.render()
				# time.sleep(w.dt/4)
				count += 1

			save_data['xi'].append(xi_arr)
			save_data['u'].append(u_opt_arr)
			save_data['vel'].append(vel_arr)
			save_data['theta_star'].append(theta_star)
			save_data['theta_predicted'].append(theta_b2/np.linalg.norm(theta_b2)*np.max(abs(theta_star)))

			w.close()

		pickle.dump(save_data, open(self.save_name + '/b2.pkl', 'wb'))
	
	# EVALUATE MISSPECIFIED OBJECTIVE FUNCTIONS
	def eval_b3(self, args):
		print("[*] EVALUATING BASELINE 3")

		save_data  = {'xi': [], 'u': [], 'vel': [], 'theta_star': [], 'theta_predicted': [], 'cost': []}

		n_eval = args.n_eval
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		gamma = 1.0
		dist_threshold = 1.5

		theta_s = np.array([[1.0, 1.0, 0.1], [0.1, -1.0, 1.0]])

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/len(theta_s))):
			THETA[idx*len(theta_s):idx*len(theta_s)+len(theta_s)] = theta_s + np.random.normal(0, 0.1, (len(theta_s), n_features))

		U = np.transpose(np.vstack((np.array([0.]*500), np.array([0.]*500), 0.5 - 1*np.random.rand(500), 0.5 - 1*np.random.rand(500))))

		U = torch.FloatTensor(np.concatenate((U, np.array([[0., 0.0, 0., 0.]]))))
		
		for _ in tqdm(range(n_eval)):
			mpc = MPC(3, 10)
			w, c1, c2 = mpc.create_world()
			if args.uniform:
				theta_star = 1 - 2*np.random.rand(n_features)
			else:
				theta_star = THETA[np.random.choice(n_samples)]
				theta_star[0] = np.clip(theta_star[0], 0, 1.5)
				theta_star[2] = np.clip(theta_star[2], 0, 1.5)

			theta_0 = np.zeros(n_features)

			theta_b3 = np.copy(theta_0)

			n_obs_steps = 9
			u_arr = []
			count = 0
			cost = 0

			xi_arr = []
			u_opt_arr = []
			vel_arr = []
			
			for idx in range(10):
				pos = torch.FloatTensor(np.array([c1.center.x, c1.center.y, c2.center.x, c2.center.y]))

				vel = torch.FloatTensor(np.array([c1.velocity.x, c1.velocity.y, c2.velocity.x, c2.velocity.y]))

				heading = torch.FloatTensor(np.array([c1.heading, c2.heading]))

				mpc.update_world(w)
				u_h = mpc.optimize_h(theta_star, args)
				u_r = mpc.optimize_r(theta_b3, count, args)

				bias = np.array([args.bias, args.bias])

				u_h += np.random.rand(2)*args.noise + bias

				u_opt = torch.FloatTensor(np.array([u_r[0], u_r[1], u_h[0], u_h[1]]))

				xi_arr.append(pos.detach().numpy())
				u_opt_arr.append(u_opt.detach().numpy())
				vel_arr.append(vel.detach().numpy())

				u_arr.append([u_r[0], u_r[1], u_h[0], u_h[1]])

				if count <= n_obs_steps:

					u_actual = torch.FloatTensor(np.copy(u_opt))

					theta_star_h = torch.eye(3)
					u_dist = []

					for id_x in range(n_features):
						for id_y in range(n_features):
							theta_star_h *= (-1)^id_x
							u_opt = self.inv_g(pos, U, vel, heading, torch.clone(torch.FloatTensor(theta_b3)), theta_star_h[id_y], g_function_eval, gamma)

							u_dist.append(torch.norm(u_opt - u_actual).detach().numpy())

					if np.array(u_dist).any() < dist_threshold:
						theta_b3 += gamma * g_function_eval(pos, u_actual, vel, heading, args).detach().numpy()

				c1.set_control(u_r[0], u_r[1])
				c2.set_control(u_h[0], u_h[1])
				w.tick()
				# w.render()
				# time.sleep(w.dt/4)
				count += 1

			save_data['xi'].append(xi_arr)
			save_data['u'].append(u_opt_arr)
			save_data['vel'].append(vel_arr)
			save_data['theta_star'].append(theta_star)
			save_data['theta_predicted'].append(theta_b3/np.linalg.norm(theta_b3)*np.max(abs(theta_star)))

			w.close()

		pickle.dump(save_data, open(self.save_name + '/b3.pkl', 'wb'))
	
	def plot(self, args):

		algo = ['ours', 'e2e', 'b1', 'b2', 'b3']
		fig, ax = plt.subplots()
		for alg_id, alg in enumerate(algo):
			data = pickle.load(open(self.save_name + '/' + alg + '.pkl', 'rb'))
			theta_star = np.array(data['theta_star'])
			for t_idx, ts in enumerate(theta_star):
				theta_star[t_idx] = ts/np.linalg.norm(ts)
				
			theta_pred = np.array(data['theta_predicted'])
			theta_err = np.mean(np.linalg.norm(theta_star - theta_pred, axis=1))
			theta_std = np.std(np.linalg.norm(theta_star - theta_pred, axis=1))
			print(alg)
			print(np.mean(np.linalg.norm(theta_star - theta_pred, axis=1)), np.std(np.linalg.norm(theta_star - theta_pred, axis=1)))

			ax.bar(alg_id, theta_err, yerr=theta_std/np.sqrt(len(theta_pred)))
		ax.set_title('uniform prior')
		ax.set_xticks(np.arange(len(algo)))
		ax.set_xticklabels(algo)

		plt.show()
		fig.savefig('g_func_exp/results/uniform_prior.png')
		fig.savefig('g_func_exp/results/uniform_prior.svg')
