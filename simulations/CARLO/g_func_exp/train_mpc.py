import numpy as np
import torch
import os, sys
import torch.optim as optim
from tqdm import tqdm
from g_func_exp.model import Model, Model_end2end, g_function_eval
from mpc_highway import MPC

class TRAIN():
	def __init__(self, args):
		self.args = args

		if not os.path.exists('g_func_exp/g_data/hidden{}'.format(args.hidden_size)):
			os.makedirs('g_func_exp/g_data/hidden{}'.format(args.hidden_size))
			
		self.save_name = 'g_func_exp/g_data/hidden{}'.format(args.hidden_size)

		self.train(args)

	def inv_g(self, x, U, vel, heading, theta0, theta_star, g, gamma):
		n_samples = len(U)
		theta_err = np.array([0.]*n_samples)
		for idx, u_h in enumerate(U):
			theta_estimate = theta0 + gamma*g(x, u_h, vel, heading, self.args)
			theta_err[idx] = torch.linalg.norm(theta_star - theta_estimate).detach().numpy()
		opt_idx = np.argmin(theta_err)
		return U[opt_idx]

	def train(self, args):
		env_dim = args.env_dim
		n_features = args.n_features
		n_samples = 3000
		gamma = 1.0

		input_size = 7
		output_size = n_features
		hidden_size = args.hidden_size
		LR = args.lr

		model = Model(input_size, output_size, hidden_size)
		# model = Model_end2end(input_size, output_size, hidden_size)

		optimizer = optim.Adam(model.parameters(), lr = LR)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.lr_gamma)

		theta_s = np.array([[1.0, 1.0, 0.1], [0.1, -1.0, 1.0]])

		THETA = np.zeros((n_samples, n_features))
		for idx in range(int(n_samples/len(theta_s))):
			THETA[idx*len(theta_s):idx*len(theta_s)+len(theta_s)] = theta_s + np.random.normal(0, 0.1, (len(theta_s), n_features))

		n_batch = args.batch_size
		EPOCH = args.n_train_steps

		for epoch in tqdm(range(EPOCH + 1)):
			u_opt_arr = torch.zeros((n_batch, env_dim*2))
			pos_arr = torch.zeros((n_batch, env_dim*2))
			theta_star_arr = torch.zeros((n_batch, n_features))
			theta0_arr = torch.zeros((n_batch, n_features))
			g_tilde = torch.zeros((n_batch, n_features))

			for idx in range(n_batch):
				c1_x = 55
				c2_x = 55
				c1_y = np.random.randint(40, 50)
				c2_y = np.random.randint(20, 35)
				
				pos = torch.FloatTensor(np.array([c1_x, c1_y, c2_x, c2_y]))

				vel = torch.FloatTensor(np.array([0, 1.5, 0, 1.5]))
				heading = torch.FloatTensor(np.array([np.pi/2, np.pi/2]))

				theta_0 = torch.FloatTensor(np.array([np.random.rand(), 1-2*np.random.rand(), np.random.rand()]))
				
				theta_star = torch.FloatTensor(THETA[np.random.choice(n_samples)])

				theta_star[0] = np.clip(theta_star[0], 0, 1.3)

				U = np.transpose(np.vstack((np.array([0.]*500), np.array([0.]*500), 0.5 - 1*np.random.rand(500), 0.5 - 1*np.random.rand(500))))

				U = torch.FloatTensor(np.concatenate((U, np.array([[0., 0., 0., 0.]]))))

				u_opt = self.inv_g(pos, U, vel, heading, theta_0, theta_star, g_function_eval, gamma)

				deltas = torch.randn_like(u_opt)*args.noise

				g_t = model.g_tilde_eval(pos, u_opt + deltas, vel, heading, args)

				u_opt_arr[idx, :] = u_opt
				pos_arr[idx, :] = pos
				theta0_arr[idx, :] = theta_0
				theta_star_arr[idx, :] = theta_star
				g_tilde[idx, :] = g_t

			err_theta = theta_star_arr - theta0_arr
			loss1 = torch.sum(torch.norm(g_tilde, dim=1)**2)
			loss2 = torch.sum(err_theta * g_tilde)

			loss = gamma*loss1 - 2*loss2

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			tqdm.write("loss at epoch {} is {}".format(epoch, loss))
			if epoch%100 == 0:
				torch.save(model.state_dict(), self.save_name + '/model_' + str(args.n_features) + 'objs_' + str(epoch))