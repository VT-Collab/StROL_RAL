import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def g_function_eval(x, u, vel, heading, args):
	g = torch.zeros((args.n_features))

	for idx in range (args.n_features):
		if idx == 0:
			x_dist = x[0] - x[2]
			y_dist_old = torch.norm(x[1] - x[3])
			y_dist_new = torch.norm(x[1] + vel[1] + u[1] - (x[3] + vel[3] + u[3]))
			y_dist = -y_dist_new + y_dist_old
			g[idx] = (y_dist/2)**2

		if idx == 1:
			new_speed = vel[3] + u[3]
			old_speed = vel[3]
			speed_diff = -new_speed + old_speed
			speed = torch.min(torch.FloatTensor([0.25, speed_diff]))
			g[idx] = (speed)**2

		if idx == 2:
			x_dist = x[0] - x[2]
			head = u[2]
			if x_dist < -8:
				head =  abs(torch.pi/2 - (heading[1] + u[2]))
			g[idx] = -1*head
	return 1*g


def limit_g(g, g_hat):
	if g.dim() == 1:
		norm_g = torch.norm(g)
		norm_g_hat = torch.norm(g_hat)
		if norm_g_hat > norm_g:
			g_hat *= norm_g/norm_g_hat
	return g_hat


class Model(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Model, self).__init__()

		self.learner = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, int(hidden_dim/2)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/4), int(hidden_dim/8)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/8), action_dim),
			nn.Tanh()
		)



	def forward(self, x):
		out = self.learner(x)
		return out
	

	def g_tilde_eval(self, x, u, vel, heading, args):

		g_hat = self.forward(torch.cat((x[2:], u[2:], vel[2:], heading[1:])))
		g = g_function_eval(x, u, vel, heading, args)
		if args.eval:
			g_hat = limit_g(g, g_hat)
		return g + 1*g_hat


class Model_end2end(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Model_end2end, self).__init__()

		self.learner = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, int(hidden_dim/2)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/4), int(hidden_dim/8)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/8), action_dim),
			nn.Tanh()
		)
		

	def forward(self, x):
		out = self.learner(x)
		return out

	def g_tilde_eval(self, x, u, vel, heading, args):
		g_hat = self.forward(torch.cat((x[2:], u[2:], vel[2:], heading[1:])))
		return g_hat