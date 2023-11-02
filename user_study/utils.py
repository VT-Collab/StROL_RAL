import numpy as np
import pickle
import torch
import time 
import os, sys
import pygame
import  socket
from scipy.interpolate import interp1d
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from model import Model_t1, Model_t2


"""Connecting and Sending commands to robot"""
class Robot():
	def __init__(self, args):
		self.args = args
		if self.args.task == 1:
			self.HOME = [0.8385, -0.0609, 0.2447, -1.5657, 0.0089, 1.5335, 1.8607]
		elif self.args.task == 2 or self.args.task == 3:
			self.HOME = [1.13894, -0.129095,-0.236155,  -1.73935, 0.566207, 2.39615, 1.49282]


	def connect2robot(self, PORT):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('172.16.0.3', PORT))
		s.listen()
		conn, addr = s.accept()
		return conn

	def send2robot(self, conn, qdot, mode, traj_name=None, limit=0.5):
		if traj_name is not None:
			if traj_name[0] == 'q':
				# print("limit increased")
				limit = 1.0
		qdot = np.asarray(qdot)
		scale = np.linalg.norm(qdot)
		if scale > limit:
			qdot *= limit/scale
		send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
		send_msg = "s," + send_msg + "," + mode + ","
		conn.send(send_msg.encode())

	def send2gripper(self, conn):
		send_msg = "o"
		conn.send(send_msg.encode())

	def listen2robot(self, conn):
		state_length = 7 + 7 + 7 + 6 + 42
		message = str(conn.recv(2048))[2:-2]
		state_str = list(message.split(","))
		for idx in range(len(state_str)):
			if state_str[idx] == "s":
				state_str = state_str[idx+1:idx+1+state_length]
				break
		try:
			state_vector = [float(item) for item in state_str]
		except ValueError:
			return None
		if len(state_vector) is not state_length:
			return None
		state_vector = np.asarray(state_vector)
		state = {}
		state["q"] = state_vector[0:7]
		state["dq"] = state_vector[7:14]
		state["tau"] = state_vector[14:21]
		state["O_F"] = state_vector[21:27]
		state["J"] = state_vector[27:].reshape((7, 6)).T
		
		# get cartesian pose
		xyz_lin, R = self.joint2pose(state_vector[0:7])
		beta = -np.arcsin(R[2,0])
		alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
		gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
		xyz_ang = [alpha, beta, gamma]
		xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
		state["x"] = np.array(xyz)
		return state

	def readState(self, conn):
		while True:
			state = self.listen2robot(conn)
			if state is not None:
				break
		return state

	def xdot2qdot(self, xdot, state):
		J_pinv = np.linalg.pinv(state["J"])
		return J_pinv @ np.asarray(xdot)

	def joint2pose(self, q):
		def RotX(q):
			return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
		def RotZ(q):
			return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
		def TransX(q, x, y, z):
			return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
		def TransZ(q, x, y, z):
			return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
		H1 = TransZ(q[0], 0, 0, 0.333)
		H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
		H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
		H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
		H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
		H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
		H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
		H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
		H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
		return H[:,3][:3], H[:,:3][:3]

	def go2home(self, conn, h=None):
		if h is None:
			home = np.copy(self.HOME)
		else:
			home = np.copy(h)
		total_time = 35.0;
		start_time = time.time()
		state = self.readState(conn)
		current_state = np.asarray(state["q"].tolist())

		# Determine distance between current location and home
		dist = np.linalg.norm(current_state - home)
		curr_time = time.time()
		action_time = time.time()
		elapsed_time = curr_time - start_time

		# If distance is over threshold then find traj home
		while dist > 0.1 and elapsed_time < total_time:
			current_state = np.asarray(state["q"].tolist())

			action_interval = curr_time - action_time
			if action_interval > 0.005:
				# Get human action
				qdot = home - current_state
				# qdot = np.clip(qdot, -0.3, 0.3)
				self.send2robot(conn, qdot, "v")
				action_time = time.time()

			state = self.readState(conn)
			dist = np.linalg.norm(current_state - home)
			curr_time = time.time()
			elapsed_time = curr_time - start_time

		# Send completion status
		if dist <= 0.02:
			return True
		elif elapsed_time >= total_time:
			return False

	def wrap_angles(self, theta):
		if theta < -np.pi:
			theta += 2*np.pi
		elif theta > np.pi:
			theta -= 2*np.pi
		else:
			theta = theta
		return theta

	# play the trajectory generated
	def play_traj(self, conn, traj, state, start_t):
		curr_t = time.time() - start_t
		x_des = traj.get(curr_t)
		x_curr = state['x']

		if x_curr[3] < 0:
			x_curr[3] += np.pi
		elif x_curr[3] > 0:
			x_curr[3] -= np.pi  

		xdot = 1*(x_des - x_curr)

		if self.args.task == 1:
			xdot[3:] = 0*xdot[3:]

		qdot = self.xdot2qdot(xdot, state)
		q_curr = state['q']

		self.send2robot(conn, qdot, mode='v')

	# record a correction
	def record_correction(self, conn, xi_c):
		state = self.readState(conn)
		pos = state['x']
		if pos[3] < 0:
			pos[3] += np.pi
		elif pos[3] > 0:
			pos[3] -= np.pi
			
		xi_c.append(pos)
		return xi_c

	# update trajectory with timestamp for execution
	def update_traj(self, traj, time):
		if self.args.task == 1:
			traj = np.hstack((traj, np.zeros((len(traj), 3))))
		xi_r = Trajectory(traj, time)
		return xi_r
	
	def play_demo(self, conn, xi):
		traj = self.update_traj(xi, 30.0)
		start_t = time.time()
		while True:
			state = self.readState(conn)
			self.play_traj(conn, traj, state, start_t)
			if time.time() - start_t > 30.0 or np.linalg.norm(state['O_F']) > 30:
				break


"""Interpolating the generated trajectory for execution on robot"""
class Trajectory(object):

	def __init__(self, xi, T):
		""" create cublic interpolators between waypoints """
		self.xi = np.asarray(xi)
		self.T = T
		self.n_waypoints = xi.shape[0]
		timesteps = np.linspace(0, self.T, self.n_waypoints)
		self.f1 = interp1d(timesteps, self.xi[:,0], kind='cubic')
		self.f2 = interp1d(timesteps, self.xi[:,1], kind='cubic')
		self.f3 = interp1d(timesteps, self.xi[:,2], kind='cubic')
		self.f4 = interp1d(timesteps, self.xi[:,3], kind='cubic')
		self.f5 = interp1d(timesteps, self.xi[:,4], kind='cubic')
		self.f6 = interp1d(timesteps, self.xi[:,5], kind='cubic')

	def get(self, t):
		""" get interpolated position """
		if t < 0:
			q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0)]
		elif t < self.T:
			q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t)]
		else:
			q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T), self.f5(self.T), self.f6(self.T)]
		return np.asarray(q)


"""Define hoystich inputs for robot"""
class Joystick(object):

	def __init__(self):
		pygame.init()
		self.gamepad = pygame.joystick.Joystick(0)
		self.gamepad.init()
		self.deadband = 0.1
		self.timeband = 0.5
		self.lastpress = time.time()

	def input(self):
		pygame.event.get()
		curr_time = time.time()
		A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
		B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
		X_pressed = self.gamepad.get_button(2) and (curr_time - self.lastpress > self.timeband)
		START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
		if A_pressed or START_pressed or B_pressed:
			self.lastpress = curr_time
		return A_pressed, B_pressed, X_pressed, START_pressed


"""Trajectory optimization for learned task parameters"""
class TrajOpt(object):

	def __init__(self, args, home, goal, state_len, waypoints=10):
		# initialize trajectory
		self.args = args
		self.n_waypoints = waypoints
		self.state_len = state_len
		self.home = home
		self.goal = goal
		self.n_joints = len(self.home)
		self.provide_demos = True
		self.xi0 = np.zeros((self.n_waypoints, self.n_joints))
		for idx in range(self.n_waypoints):
			self.xi0[idx, :] = self.home + idx/(self.n_waypoints - 1.0) * (self.goal - self.home)
		self.xi0 = self.xi0.reshape(-1)

		# create start constraint and action constraint
		self.B = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))
		for idx in range(self.n_joints):
			self.B[idx,idx] = 1
		self.G = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))
		for idx in range(self.n_joints):
			self.G[self.n_joints-idx-1,self.n_waypoints*self.n_joints-idx-1] = 1
		self.lincon = LinearConstraint(self.B, self.home, self.home)
		self.lincon2 = LinearConstraint(self.G, self.goal, self.goal)
		
		self.nonlincon_lin = NonlinearConstraint(self.nl_function_lin, -0.05, 0.05)
		self.nonlincon_ang = NonlinearConstraint(self.nl_function_ang, -0.1, 0.1)


	# each action cannot move more than 1 unit
	def nl_function_lin(self, xi):
		xi = xi.reshape(self.n_waypoints, self.n_joints)
		actions = xi[1:, :3] - xi[:-1, :3]
		return actions.reshape(-1)

	def nl_function_ang(self, xi):
		xi = xi.reshape(self.n_waypoints, self.n_joints)
		actions = xi[1:, 3:] - xi[:-1, 3:]
		return actions.reshape(-1)
	
	# trajectory reward function
	def reward(self, xi, theta):
		self.laptop = np.array([0.35, 0.1, 0.0])
		R1 = 0
		R2 = 0
		for idx in range(len(xi)):
			R1 -= np.linalg.norm(xi[idx, :2] - self.laptop[:2])
			R2 -= np.linalg.norm(self.goal - xi[idx])
		R1 *= theta[0]
		R2 *= theta[1]

		return R1 + R2
		

	def Phi(self, states):
		phi = np.zeros(self.args.n_features)
		for idx in range (self.args.n_features):
			if idx == 0:
				for s_idx in range (self.n_waypoints):
					dist = np.linalg.norm(self.laptop[:2] - states[s_idx, :2])
					phi[idx] -= dist
			if idx == 1:
				for s_idx in range (self.n_waypoints):
					phi[idx] -= np.linalg.norm(self.goal[:3] - states[s_idx, :3])
			if idx == 2:
				for s_idx in range (self.n_waypoints):
					phi[idx] -= np.linalg.norm(states[[s_idx, 2]])
			if idx == 3:
				for s_idx in range(self.n_waypoints):
					phi[idx] -= np.linalg.norm(states[s_idx, 3:] - self.orientation)
		return phi

	# trajectory cost function
	def trajcost(self, xi):
		xi = xi.reshape(self.n_waypoints, self.n_joints)
		states = np.zeros((self.n_waypoints, self.state_len))
		for idx in range(self.n_waypoints):
			states[idx, :] = xi[idx,:]
		states = states

		R = self.weights @ self.Phi(states)
		return -R

	# run the optimizer
	def optimize(self, weights=None, context=None, method='SLSQP'):
		self.goal = np.copy(context)
		self.laptop = np.array([0.35, 0.1, 0.0])
		self.orientation = np.array([0.0, 0.0, 0.0])
		self.weights = weights
		start_t = time.time()
		if self.args.task != 1:
			res = minimize(self.trajcost, self.xi0, method=method, constraints={self.lincon, self.nonlincon_lin, self.nonlincon_ang}, options={'eps': 1e-3, 'maxiter': 2500})
		else:
			res = minimize(self.trajcost, self.xi0, method=method, constraints={self.lincon, self.nonlincon_lin}, options={'eps': 1e-3, 'maxiter': 2500})
		xi = res.x.reshape(self.n_waypoints, self.n_joints)
		states = np.zeros((self.n_waypoints, self.state_len))
		for idx in range(self.n_waypoints):
			xi[idx, 0] = np.clip(xi[idx, 0], 0.2, 0.75) 
			xi[idx, 1] = np.clip(xi[idx, 1], -0.5, 0.5) 
			xi[idx, 2] = np.clip(xi[idx, 2], 0.05, 0.6)
			states[idx, :] = xi[idx, :]
		return states, res, time.time() - start_t


class Learner():
	def __init__(self, args):
		# define environment and algorithm parameters
		self.args = args
		self.gamma = 1.0
		self.laptop = np.array([0.35, 0.1, 0.0])
		self.orientation = np.array([0.0, 0.0, 0.0])
		self.goal = np.array([0.7, -0.45, 0.1])

		# initialize model for Ours
		self.input_size = args.env_dim +  args.env_dim
		self.output_size = args.n_features
		self.hidden_size = args.hidden_size
		
		if self.args.task == 1:
			self.model = Model_t1(self.input_size, self.output_size, self.hidden_size)
			self.model.load_state_dict(torch.load('g_data/model_t1'))

		elif self.args.task == 2 or self.args.task == 3:
			self.model = Model_t2(self.input_size, self.output_size, self.hidden_size)
			self.model.load_state_dict(torch.load('g_data/model_t2'))

	
	def cost(self, x, u, features, theta):
		cost = 0.
		for idx, feat in enumerate(features):
			if idx == 0:
				cost += theta[idx] * np.linalg.norm(x.detach().numpy()[:2] - feat.detach().numpy()[:2])
			if idx == 1:
				cost += theta[idx] * np.linalg.norm(x.detach().numpy()[:3] - feat.detach().numpy()[:3])
			if idx == 2:
				cost += theta[idx] * x.detach().numpy()[2]
			if idx == 3:
				cost += theta[idx] * np.linalg.norm(x.detach().numpy()[3:] - feat.detach().numpy())
		return cost


	def boltzmann_model(self, x, U, features, theta, beta=5.0):
		P = np.array([0.] * len(U))
		for idx, u in enumerate(U):
			cost = self.cost(x, u, features, theta)
			P[idx] = np.exp(-beta * cost)
		P /=np.sum(P)
		idx_star = np.argmax(P)
		return U[idx_star]


	def update_theta(self, theta, corr_id, goal):
		theta = torch.FloatTensor(theta)
		
		# load correction provided
		xi_c = pickle.load(open('data/user_{}/task_{}/{}/corrections/corr_{}.pkl'.format(self.args.user, self.args.task, self.args.alg, corr_id), 'rb'))

		xi_c = np.array(xi_c)

		# for each timestep in correction, update theta with gicen algorithm
		for idx in range(len(xi_c) - 1):
			self.table = np.array([xi_c[idx, 0], xi_c[idx, 1], 0.0])
			if self.args.task == 1:
				features = torch.FloatTensor(np.array([self.laptop, self.goal, self.table]))
			else: 
				features = torch.FloatTensor(np.array([self.laptop, self.goal, self.table, self.orientation]))

			pos = torch.FloatTensor(xi_c[idx, :])
			u = torch.FloatTensor(xi_c[idx+1, :] - xi_c[idx, :])

			if self.args.alg == 'strol':
				theta += self.gamma * self.model.g_tilde_eval(pos, 3*u, features, self.args)[-1]
			elif self.args.alg == 'oat':
				g_func = self.model.g_function_eval(pos, 3*u, features, self.args)[-1]
				update_idx = np.argmax(abs(g_func.detach().numpy()))
				theta[update_idx] += self.gamma * g_func[update_idx]
			elif self.args.alg == 'mof':
				dist_threshold = 2.0
				U = torch.FloatTensor(3 * np.max(abs(u.detach().numpy()))* (1 - 2*np.random.rand(100, self.args.env_dim)))
				theta_star_h = torch.eye(self.args.env_dim)
				u_dist = []
				for id_x in range (self.args.n_features):
					for id_y in range(self.args.n_features):
						u_opt = self.boltzmann_model(pos, U, features, theta_star_h[id_y], 5.)
					if self.args.task == 1:
						u_dist.append(torch.norm(u_opt - 3*u[:3]).detach().numpy())
					else:
						u_dist.append(torch.norm(u_opt - 3*u).detach().numpy())
				if (np.array(u_dist) < dist_threshold).any():
					g_func = self.model.g_function_eval(pos, 3*u, features, self.args)[-1]
					theta += self.gamma * g_func


		return theta.detach().numpy()

		