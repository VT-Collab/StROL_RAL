import numpy as np
import pickle
from utils import Robot, TrajOpt, Learner
import time


class Task_1():
    def __init__(self, args):
        self.args = args
        self.robot = Robot(args)
        self.learner = Learner(args)

        PORT = 8080
        print('[*] Connecting to low-level controller...')
        self.conn = self.robot.connect2robot(PORT)
        print("Connection Established")
        print("Returning Home")
        self.robot.go2home(self.conn)

        self.save_data = {'theta_learned': [], 'theta_star': [], 'robot_traj': [], 'corrections': [], 'traj_learned': [], 'traj_star':[]}

    def test(self):
        args = self.args
        
        # initialize the task parameters
        theta_s = np.array([-1.5, 0.8, 0.8])
        theta = 1 - 2*np.random.rand(3)
        print(theta)

        # define initial state and goal
        state = self.robot.readState(self.conn)
        start_pos = np.array(state['x'][:3])
        
        goal = np.array([0.7, -0.45, 0.1])
        
        # show a demo to user and get initial robot trajectory
        trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=20)
        xi_demo, _, _ = trajopt.optimize(theta_s, goal)
        if args.demo:
            print("[*] Press ENTER to Start Demo")
            input()
            self.robot.play_demo(self.conn, xi_demo)
            self.robot.go2home(self.conn)
        xi_init, _, _ = trajopt.optimize(theta, goal)
        
        

        traj_time = 45.0
        traj = self.robot.update_traj(xi_init, traj_time)

        corr_id = 0
        print("[*] Press ENTER to Start")
        input()

        start_t = time.time()
        traj_start_time = time.time()
        time_elapsed = 0.0
        
        XI_R = []
        XI_C = []
        THETA_L = []
        # main loop for the task
        while True:
            state = self.robot.readState(self.conn)
            ef_force = np.linalg.norm(state['O_F'])


            # execute trajectory if there is no human input
            if ef_force <= 15:
                self.robot.play_traj(self.conn, traj, state, start_t)
                time_elapsed += time.time() - traj_start_time
                XI_R.append(state['x'])
            else:
                xi_c = []
                corr_id += 1
                max_force = ef_force
                time_elapsed += time.time() - start_t
                traj_time -= time.time() - start_t
                if traj_time < 5:
                    traj_time = 5.0

                # if human input, record and save correction
                while ef_force > 15:
                    state = self.robot.readState(self.conn)
                    ef_force = np.linalg.norm(state['O_F'])
                    if ef_force > max_force:

                        max_force = ef_force
                        xi_c = self.robot.record_correction(self.conn, xi_c)
                XI_C.append(xi_c)
                pickle.dump(xi_c, open('data/user_{}/task_{}/{}/corrections/corr_{}.pkl'.format(args.user, args.task, args.alg, corr_id), 'wb'))

                # update reward weights (theta) of the robot
                theta = self.learner.update_theta(theta, corr_id, goal)
                THETA_L.append(theta)
                print(theta)

                # recompute the trajectory based on new reward weights
                state = self.robot.readState(self.conn)
                start_pos = state['x'][:3]
                trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=15)
                xi, _, _ = trajopt.optimize(theta, goal)

                traj = self.robot.update_traj(xi, traj_time)
                print('[*] Press ENTER to Start')
                input()
                start_t = time.time()

            traj_start_time = time.time()
            # if trajectory plays for more than a minute, end the interaction
            if time_elapsed >= 60:
                print(time_elapsed)
                self.save_data['theta_learned'].append(THETA_L)
                self.save_data['theta_star'].append(theta_s)
                self.save_data['robot_traj'].append(XI_R)
                self.save_data['corrections'].append(XI_C)
                break
        
        # send the robot to home and play the learned trajectory based on learned reward weights
        print("[*] Trajectory Ended !!")
        print("[*] Press enter to start showing the demo")
        input()
        self.robot.go2home(self.conn)
        start_pos = self.robot.readState(self.conn)['x'][:3]
        trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=20)
        xi, _, _ = trajopt.optimize(theta, goal)
        xi_star, _, _ = trajopt.optimize(theta_s, goal)

        self.save_data['traj_learned'].append(xi)
        self.save_data['traj_star'].append(xi_star)

        self.robot.play_demo(self.conn, xi)
        # save the data for the interaction
        pickle.dump(self.save_data, open('data/user_{}/task_{}/{}/data.pkl'.format(self.args.user, self.args.task, self.args.alg), 'wb'))

        

class Task_2():
    def __init__(self, args):
        self.args = args
        self.robot = Robot(args)
        self.learner = Learner(args)

        PORT = 8080
        print('[*] Connecting to low-level controller...')
        self.conn = self.robot.connect2robot(PORT)
        print("Connection Established")
        print("Returning Home")
        self.robot.go2home(self.conn)

        self.save_data = {'theta_learned': [], 'theta_star': [], 'robot_traj': [], 'corrections': [], 'traj_learned': [], 'traj_star':[]}

    def test(self):
        args = self.args
        
        # initialize the task parameters
        theta_s = np.array([-0.8, 0.8, 0.8, 0.8])
        theta = 1 - 2*np.random.rand(4)
        print(theta)

        # define initial state and goal
        state = self.robot.readState(self.conn)
        start_pos = np.array(state['x'])
        if start_pos[3] < 0:
            start_pos[3] += np.pi
        elif start_pos[3] > 0:
            start_pos[3] -= np.pi

        goal = np.array([0.7, -0.45, 0.1, start_pos[3], start_pos[4], start_pos[5]])

        # get initial robot trajectory
        trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=20)
        xi_demo, _, _ = trajopt.optimize(theta_s, goal)
        if args.demo:
            print("[*] Press ENTER to Start Demo")
            input()
            self.robot.play_demo(self.conn, xi_demo)
            self.robot.go2home(self.conn)
        xi_init, _, _ = trajopt.optimize(theta, goal)


        traj_time = 45.0
        traj = self.robot.update_traj(xi_init, traj_time)

        corr_id = 0
        print("[*] Press ENTER to Start")
        input()
       
        start_t = time.time()
        traj_start_time = time.time()
        time_elapsed = 0.0
        
        XI_R = []
        XI_C = []
        THETA_L = []
        # main loop for the task
        while True:
            state = self.robot.readState(self.conn)
            ef_force = np.linalg.norm(state['O_F'])

            # execute trajectory if there is no human input
            if ef_force <= 15:
                self.robot.play_traj(self.conn, traj, state, start_t)
                time_elapsed += time.time() - traj_start_time
                XI_R.append(state['x'])
            else:
                xi_c = []
                corr_id += 1
                max_force = ef_force
                traj_time -= time.time() - start_t
                if traj_time < 5:
                    traj_time = 5.0

                # if human input, record and save correction
                while ef_force > 15:
                    state = self.robot.readState(self.conn)
                    ef_force = np.linalg.norm(state['O_F'])
                    if ef_force > max_force:
                        max_force = ef_force
                        xi_c = self.robot.record_correction(self.conn, xi_c)
                XI_C.append(xi_c)
                pickle.dump(xi_c, open('data/user_{}/task_{}/{}/corrections/corr_{}.pkl'.format(args.user, args.task, args.alg, corr_id), 'wb'))

                # update reward weights (theta) of the robot
                theta = self.learner.update_theta(theta, corr_id, goal)
                THETA_L.append(theta)
                print(theta)

                # recompute the trajectory based on new reward weights
                state = self.robot.readState(self.conn)
                start_pos = state['x']
                if start_pos[3] < 0:
                    start_pos[3] += np.pi
                elif start_pos[3] > 0:
                    start_pos[3] -= np.pi

                trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=15)
                xi, _, _ = trajopt.optimize(theta, goal)

                traj = self.robot.update_traj(xi, traj_time)
                print('[*] Press ENTER to Start')
                input()
                start_t = time.time()

            traj_start_time = time.time()
            if time_elapsed >= 60:
                print(time_elapsed)
                self.save_data['theta_learned'].append(THETA_L)
                self.save_data['theta_star'].append(theta_s)
                self.save_data['robot_traj'].append(XI_R)
                self.save_data['corrections'].append(XI_C)
                break

        # send the robot to home and play the learned trajectory based on learned reward weights
        print("[*] Trajectory Ended !!")
        print("[*] Press enter to start showing the demo")
        input()
        self.robot.go2home(self.conn)
        start_pos = self.robot.readState(self.conn)['x']
        trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=20)
        if start_pos[3] < 0:
            start_pos[3] += np.pi
        elif start_pos[3] > 0:
            start_pos[3] -= np.pi
        xi, _, _ = trajopt.optimize(theta, goal)
        xi_star, _, _ = trajopt.optimize(theta_s, goal)

        self.save_data['traj_learned'].append(xi)
        self.save_data['traj_star'].append(xi_star)

        self.robot.play_demo(self.conn, xi)
        # save the data for the interaction
        pickle.dump(self.save_data, open('data/user_{}/task_{}/{}/data.pkl'.format(self.args.user, self.args.task, self.args.alg), 'wb'))

        

class Task_3():
    def __init__(self, args):
        self.args = args
        self.robot = Robot(args)
        self.learner = Learner(args)

        PORT = 8080
        print('[*] Connecting to low-level controller...')
        self.conn = self.robot.connect2robot(PORT)
        print("Connection Established")
        print("Returning Home")
        self.robot.go2home(self.conn)

        self.save_data = {'theta_learned': [], 'theta_star': [], 'robot_traj': [], 'corrections': [], 'traj_learned': [], 'traj_star':[]}

    def test(self):
        args = self.args
        
        # initialize the task parameters
        theta_s = np.array([-0.8, -0.8, -0.8, 0.8])
        theta = 1 - 2*np.random.rand(4)
        print(theta)

        # define initial state and goal
        state = self.robot.readState(self.conn)
        start_pos = np.array(state['x'])
        if start_pos[3] < 0:
            start_pos[3] += np.pi
        elif start_pos[3] > 0:
            start_pos[3] -= np.pi

        goal = np.array([0.7, -0.45, 0.1, start_pos[3], start_pos[4], start_pos[5]])

        # get initial robot trajectory
        trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=20)
        xi_demo, _, _ = trajopt.optimize(theta_s, goal)
        if args.demo:
            print("[*] Press ENTER to Start Demo")
            input()
            self.robot.play_demo(self.conn, xi_demo)
            self.robot.go2home(self.conn)
        xi_init, _, _ = trajopt.optimize(theta, goal)
        self.robot.go2home(self.conn)

        traj_time = 45.0
        traj = self.robot.update_traj(xi_init, traj_time)

        corr_id = 0
        print("[*] Press ENTER to Start")
        input()
        
        start_t = time.time()
        traj_start_time = time.time()
        time_elapsed = 0.0
        
        XI_R = []
        XI_C = []
        THETA_L = []
        # main loop for the task
        while True:
            state = self.robot.readState(self.conn)
            ef_force = np.linalg.norm(state['O_F'])

            # execute trajectory if there is no human input
            if ef_force <= 15:
                self.robot.play_traj(self.conn, traj, state, start_t)
                time_elapsed += time.time() - traj_start_time
                XI_R.append(state['x'])
            else:
                xi_c = []
                corr_id += 1
                max_force = ef_force
                traj_time -= time.time() - start_t
                if traj_time < 5:
                    traj_time = 5.0

                # if human input, record and save correction
                while ef_force > 15:
                    state = self.robot.readState(self.conn)
                    ef_force = np.linalg.norm(state['O_F'])
                    if ef_force > max_force:
                        max_force = ef_force
                        xi_c = self.robot.record_correction(self.conn, xi_c)
                XI_C.append(xi_c)
                pickle.dump(xi_c, open('data/user_{}/task_{}/{}/corrections/corr_{}.pkl'.format(args.user, args.task, args.alg, corr_id), 'wb'))

                # update reward weights (theta) of the robot
                theta = self.learner.update_theta(theta, corr_id, goal)
                THETA_L.append(theta)
                print(theta)

                # recompute the trajectory based on new reward weights
                state = self.robot.readState(self.conn)
                start_pos = state['x']
                if start_pos[3] < 0:
                    start_pos[3] += np.pi
                elif start_pos[3] > 0:
                    start_pos[3] -= np.pi
                    
                trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=15)
                xi, _, _ = trajopt.optimize(theta, goal)

                traj = self.robot.update_traj(xi, traj_time)
                print('[*] Press ENTER to Start')
                input()
                start_t = time.time()

            traj_start_time = time.time()
            if time_elapsed >= 60:
                print(time_elapsed)
                self.save_data['theta_learned'].append(THETA_L)
                self.save_data['theta_star'].append(theta_s)
                self.save_data['robot_traj'].append(XI_R)
                self.save_data['corrections'].append(XI_C)
                break

        # send the robot to home and play the learned trajectory based on learned reward weights
        print("[*] Trajectory Ended !!")
        print("[*] Press enter to start showing the demo")
        input()
        self.robot.go2home(self.conn)
        start_pos = self.robot.readState(self.conn)['x']
        trajopt = TrajOpt(args, start_pos, goal, args.env_dim, waypoints=20)
        if start_pos[3] < 0:
            start_pos[3] += np.pi
        elif start_pos[3] > 0:
            start_pos[3] -= np.pi
        xi, _, _ = trajopt.optimize(theta, goal)
        xi_star, _, _ = trajopt.optimize(theta_s, goal)

        self.save_data['traj_learned'].append(xi)
        self.save_data['traj_star'].append(xi_star)

        self.robot.play_demo(self.conn, xi)
        # save the data for the interaction
        pickle.dump(self.save_data, open('data/user_{}/task_{}/{}/data.pkl'.format(self.args.user, self.args.task, self.args.alg), 'wb'))
        