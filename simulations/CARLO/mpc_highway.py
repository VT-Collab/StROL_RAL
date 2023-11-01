import numpy as np
from scipy.optimize import minimize, LinearConstraint
import copy
from world import World
from agents import Car, RectangleBuilding, Painting
from geometry import Point
import time
import torch

def world():
    w = World(dt = 1.0, width = 120, height = 120, ppm = 5)
    w.add(Painting(Point(25, 60), Point(50, 150), 'gray80'))
    w.add(Painting(Point(95, 60), Point(50, 150), 'gray80'))
    w.add(RectangleBuilding(Point(20, 60), Point(48, 150)))
    w.add(RectangleBuilding(Point(100, 60), Point(48, 150)))
    for idx in range(0, 121, 10):
        w.add(Painting(Point(60, idx), Point(0.5, 3), 'white'))
    # c1_x = np.random.uniform(52, 68)
    # c2_x = np.random.uniform(52, 68)
    c1_x = 55
    c1_y = np.random.randint(40, 50)
    c1 = Car(Point(c1_x, c1_y), np.pi/2, 'orange')
    c1.velocity = Point(0, 1.5)
    c1.max_speed = 1.5
    c1.min_speed = 0.75
    w.add(c1)

    c2_x = 55
    c2_y = np.random.randint(20, 25)
    # while c1_y - c2_y < 15 or c1_y - c2_y > 30:
    #     c2_y = np.random.randint(10, 75)
    c2 = Car(Point(c2_x, c2_y), np.pi/2, 'blue')
    c2.velocity = Point(0, 1.5)
    c2.max_speed = 1.75
    c2.min_speed = 0
    w.add(c2)
    # w.render()
    # time.sleep(2)
    return w, c1, c2

class MPC:

    def __init__(self, horizon, max_iter):
        self.horizon = horizon
        self.max_iter = max_iter
        self.world = None
        self.agents = None
        self.robot_actions = None
        self.human_actions = None
        self.control_inputs_r = np.array([0.0]*self.horizon*2)
        self.control_inputs_h = np.array([0.0]*self.horizon*2)
        self.robot_const = LinearConstraint(np.eye(self.horizon*2), -0.5, 0.5)
        self.human_const = LinearConstraint(np.eye(self.horizon*2), -0.5, 0.5)
        self.heading = None
        self.dist_to_human = None
        self.block_human = None

    def create_world(self):
        w, c1, c2 = world()
        return w, c1, c2

    def update_world(self, world):
        self.world = world
        self.reset()

    def reset(self):
        self.agents = copy.deepcopy(self.world.dynamic_agents)

    def optimize_h(self, theta, args):
        self.theta_h = theta
        self.args = args
        res = minimize(self.human_cost, self.control_inputs_h, method='SLSQP', constraints=self.human_const, options={'eps': 1e-3, 'maxiter': self.max_iter})
        self.human_actions = res.x.reshape(self.horizon, 2)
        self.human_actions = np.clip(self.human_actions, -0.5, 0.5)
        return np.mean(self.human_actions[:2], axis=0)

    def optimize_r(self, theta, count, args):
        self.count = count
        self.theta_r = theta
        self.args = args
        res = minimize(self.robot_cost, self.control_inputs_r, method='SLSQP', constraints=self.robot_const, options={'eps': 1e-3, 'maxiter': self.max_iter})
        self.robot_actions = res.x.reshape(self.horizon, 2)
        self.robot_actions = np.clip(self.robot_actions, -0.5, 0.5)
        return np.mean(self.robot_actions, axis=0)


    def robot_cost(self, u_arr):
        cost = 0.
        c1 = self.agents[0]
        if self.theta_r[0] > 0.7 and self.theta_r[1] > 0.7 and self.theta_r[2] < 0.3 and self.count >= 9:
            for act in range(self.horizon):
                u = u_arr[act:act+2]
                x = np.array([c1.center.x, c1.center.y])
                vel = np.array([c1.velocity.x, c1.velocity.y])
                heading = np.array([c1.heading])

                head = u[0]
                if x[0] > 63 or x[0] < 53:
                    head = abs(np.pi/2 - (heading[0] + u[0]))
                else:
                    cost -= u[1]
                cost += head
            return cost
        return 0



    def human_cost(self, u_arr):
        cost = 0.
        c1 = self.agents[0]
        c2 = self.agents[1]
        
        for act in range (self.horizon):
            u = np.concatenate((np.array([0.0, 1.5]), u_arr[act:act+2]))
            x = np.array([c1.center.x, c1.center.y, c2.center.x, c2.center.y])
            vel = np.array([c1.velocity.x, c1.velocity.y, c2.velocity.x, c2.velocity.y])
            heading = np.array([c1.heading, c2.heading])
            g = self.g_function(x, u, vel, heading)

            cost += self.theta_h @ g
       
            c1.set_control(u[0], u[1])
            c2.set_control(u[2], u[3])
            c1.tick(self.world.dt)
            c2.tick(self.world.dt)
        self.reset()
        return cost



    def g_function(self, x, u, vel, heading):
        g = np.zeros((len(self.theta_h)))
        
        for idx in range (len(self.theta_h)):
            if idx == 0:
                y_dist_old = np.linalg.norm(x[1] - x[3])
                y_dist_new = np.linalg.norm(x[1] + vel[1] + u[1] - (x[3] + vel[3] + u[3]))
                y_dist = y_dist_new - y_dist_old
                g[idx] = (y_dist/2)**2

            if idx == 1:
                new_speed = vel[3] + u[3]
                old_speed = vel[3]
                speed_diff = new_speed - old_speed
                speed = np.min([0.25, speed_diff])
                g[idx] = (speed)*abs(speed)

            if idx == 2:
                head = u[2]
                if x[2] > 63 or x[2] < 53:
                    head =  abs(np.pi/2 - (heading[1] + u[2]))
                g[idx] = 1*head
        return 1*g



"""
Minimize dist and minimize speed --> theta = (+ve, +ve, 0)
Final THETA values :--> (1.0, 1.0, 0.1)
Maximize dist and maximize speed --> theta = (0, -ve, +ve)
Final THETA values :--> (0.1, -1.0, 1.0)
"""
