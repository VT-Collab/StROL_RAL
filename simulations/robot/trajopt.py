from cmath import nan
import numpy as np
import torch
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import pickle

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
            self.xi0[idx,:] = self.home + idx/(self.n_waypoints - 1.0) * (self.goal - self.home)
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


    # each action cannot move more than 1 unit
    def nl_function_lin(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_joints)
        actions = xi[1:, :3] - xi[:-1, :3]
        return actions.reshape(-1)
    
    # trajectory reward function
    def reward(self, xi, theta):
        self.laptop = np.array([0.35, 0.1, 0.0])
        self.context = np.aray(0.7, -0.45, 0.1)
        R1 = 0
        R2 = 0
        for idx in range(len(xi)):
            R1 -= np.linalg.norm(xi[idx, :2] - self.laptop[:2])
            R2 -= np.linalg.norm(self.context - xi[idx])
        R1 *= theta[0]
        R2 *= theta[1]

        return R1 + R2
        

    # true reward 
    def Phi(self, states):
        phi = np.zeros(self.args.n_features)
        for idx in range (self.args.n_features):
            if idx == 0:
                for s_idx in range (self.n_waypoints):
                    phi[idx] -= np.linalg.norm(self.laptop[:2] - states[s_idx, :2])
            if idx == 1:
                for s_idx in range (self.n_waypoints):
                    # phi[idx] -= states[s_idx, 2]
                    phi[idx] -= np.linalg.norm(self.context - states[s_idx])
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
    def optimize(self, reward_model=None, context=None, method='SLSQP'):
        self.context = np.copy(context)
        self.goal = np.copy(context)
        self.laptop = np.array([0.35, 0.1, 0.0])
        self.weights = reward_model
        start_t = time.time()
        res = minimize(self.trajcost, self.xi0, method=method, constraints={self.lincon, self.nonlincon_lin}, options={'eps': 1e-3, 'maxiter': 2500})
        
        xi = res.x.reshape(self.n_waypoints, self.n_joints)
        states = np.zeros((self.n_waypoints, self.state_len))
        for idx in range(self.n_waypoints):
            xi[idx, 2] = np.clip(xi[idx, 2], 0.0, 0.8) 
            states[idx, :] = xi[idx, :]
        return states, res, time.time() - start_t


