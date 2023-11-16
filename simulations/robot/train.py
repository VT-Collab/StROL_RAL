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
			bias = torch.FloatTensor([args.bias, args.bias, args.bias])
			deltas = torch.randn_like(u_opt_arr)*args.noise + bias

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
