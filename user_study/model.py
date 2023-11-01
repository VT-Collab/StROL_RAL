import torch
import torch.nn as nn
import torch.optim as optim

def limit_g(g, g_hat):
	if g.dim() == 1:
		norm_g = torch.norm(g)
		norm_g_hat = torch.norm(g_hat)
		if norm_g_hat > norm_g:
			g_hat *= norm_g/norm_g_hat
	return g_hat

class Model_t1(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Model_t1, self).__init__()

		self.learner = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, int(hidden_dim/2)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/4), action_dim),
			nn.Tanh()
		)

	def g_function_eval(self, x, u, objs_arr, args):
		objs = torch.reshape(objs_arr, (args.n_features, args.env_dim))
		g = torch.zeros((1, args.n_features))
		for idx, obj in enumerate(objs):
			if idx == 0:
				g[:, idx] = torch.norm(x[:2] - obj[:2]) - torch.norm(x[:2] + u[:2] - obj[:2])
			elif idx == 1:
				g[:, idx] = torch.norm(x[:3] - obj) - torch.norm(x[:3] + u[:3] - obj)
			elif idx == 2:
				g[:, idx] = torch.norm(x[2] - obj[2]) - torch.norm(x[2] + u[2] - obj[2])
		return g


	def forward(self, x):
		out = self.learner(x)
		return out

	def g_tilde_eval(self, x, u, objs, args):
		x = x[:3]
		u = u[:3]
		g_hat = self.forward(torch.cat((x,u)))
		g = self.g_function_eval(x, u, objs, args)
		if args.eval:
			g_hat = limit_g(g, g_hat)
		return g + 0.25*g_hat


class Model_t2(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Model_t2, self).__init__()

		self.learner = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, int(hidden_dim/2)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
			nn.ReLU(),
			nn.Linear(int(hidden_dim/4), action_dim),
			nn.Tanh()
		)

	def g_function_eval(self, x, u, objs_arr, args):
		objs = torch.reshape(objs_arr, (args.n_features, int(args.env_dim/2)))
		g = torch.zeros((1, args.n_features))
		for idx, obj in enumerate(objs):
			if idx == 0:
				g[:, idx] = torch.norm(x[:2] - obj[:2]) - torch.norm(x[:2] + u[:2] - obj[:2])
			elif idx == 1:
				g[:, idx] = torch.norm(x[:3] - obj) - torch.norm(x[:3] + u[:3] - obj)
			elif idx == 2:
				g[:, idx] = torch.norm(x[2] - obj[2]) - torch.norm(x[2] + u[2] - obj[2])
			elif idx == 3:
				g[:, idx] = torch.norm(x[3:] - obj) - torch.norm(x[3:] + u[3:] - obj)
		return g

	def forward(self, x):
		out = self.learner(x)
		return out


	def g_tilde_eval(self, x, u, objs, args):
		g_hat = self.forward(torch.cat((x,u)))
		g = self.g_function_eval(x, u, objs, args)
		if args.eval:
			g_hat = limit_g(g, g_hat)
		return g + 1.0*g_hat
		