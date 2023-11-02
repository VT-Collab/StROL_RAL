# StROL: Stabilized and Robust Online Learning from Humans

This repository provides our implementation of StROL in two different simulation environments and on a 7-DoF Franka Emika Panda robot arm.

## Dependencies
You need to have the following libraries with [Python3](https://www.python.org/):

- [Numpy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [PyTorch](https://pytorch.org/)
- [Tkinter](https://wiki.python.org/moin/TkInter)
- [pygame](https://www.pygame.org/news)
- [tqdm](https://tqdm.github.io/)

## Installation
To install StROL, clone the repo using
```bash
git clone https://github.com/VT-Collab/StROL.git
```


## Implementation of StROL
In StROL, we modify the learning dynamics of the system to incorporate a correction term in order to make learning from noisy and suboptimal actions more robust.

$$
\tilde g = g + \hat g
$$

The block diagram below shows the methodology used to generate the dataset using the given online learning rule $g$ and use that dataset to train the correction term $\hat g$

<center>
    <img src="./figs/block_diagram.png" alt="StROL_Framework" style="zoom:33%;" />
    <br>
    <div align="center">
        Figure 1: Framework of offline training of StROL.
    </div>
</center>

For our implementation we formulate the correction term $\hat g$ as a fully connected neural network.

Next, we explain the implementation of StROL in different simulation environments. We define the online learning rule $g$ and discuss the hyperparameters used for training.

### Highway Environment
In this setting, a robot car (a car controlled by a robot) is driving in front of a human car (a car controlled and driven by a human). Both the cars start in the left lane on a two-lane highway. The action and state space for both the cars is 2-dimensional, i.e. $x, u \in \mathbb{R}^2$. The goal of the robot car is to minimize the distance travelled and avoid any collisions with the human car. In this 2-lane highway setting, we assume two possible priors: (a) the human car will change lanes and pass the robot from the right and (b) the human car will tailgate the robot car till it gives way to the robot car. Note that prior (b), the human car does not care about the minimum distance from the robot car but wants to avoid collisions.

Below, we define the features and the online learning rule $g$ used for getting the above described behavior of the robot and the human. 

#### Features
- Distance between the cars $d = x_{\mathcal{R}} - x_{\mathcal{H}}$
- Speed of the robot car $v$
- Heading direction of the human car $h$

The learning rule of the robot $g$ is defined as the change in the features at each timestep

$$
g = [d^{t+1} - d^t, \max(0.25, v^{t+1} - v^t), -h]
$$

For the Highway environment, move to the corresponding folder using
```bash
cd simulations/CARLO
```

To train a the correction term $\hat g$, run the following command
```bash
python3 test_ours.py --train
```

We provide a trained model for the environment with $10\%$ noise and $0$ bias in the human actions. You can test the performance of this pre-trained model by running the following command
```bash
python3 test_ours.py --eval
```

This script will run the evaluation script for the Highway environment for StROL and the baselines and save a plot for the performance of the different approached.

In order to test the trained model with different noise and bias levels, you can provide the noise using `--noise` and `--bias` arguments respectively.
The full list of arguments for training and testing and their default values can be seen in `test_ours.py`

### Robot Environment
In this environment, a simulated human is trying to convey thier task preferences to the robot. The action and state spaces in this environment are both 3-dimensional, i.e. $x, u \in \mathbb{R}^3$. For training, the robot is randomly initialized in the environment and the simulated human provides corrections in order to convey their task preferences. The environment consists of two objects --- a plate and a cup. We model the prior in this environment as a bimodal distribution where the human can teach the robot either (a) to move to the cup and avoid the plate or (b) to move to the plate while avoiding the cup.

The features used in this environment and the robot's original learning dynamics $g$ are defined below.

#### Features
- 2-D distance of the robot's end-effector from the plate $d_p$
- 3-D distancs of the robot's end-effector from the cup $d_c$

The learning rule $g$ for the robot is defined as

$$
g = [d_p^{t+1} - d_p^t, d_c^{t+1} - d_c^t]
$$

Let $\theta = \{\theta_p, \theta_c\}$ be the reward parameters. The reward function of the task is defined as

$$
\mathcal{R}(\theta) = \theta_p \cdot d_p + \theta_c \cdot d_c
$$

For training the $\hat g$ for the Robot environment, move to the corresponding folder using
```bash
cd simulations/robot
```

And then run 
```bash
python3 test_ours.py --train
```
This will train $\hat g$ to expand the basins of attraction and enable learning from noisy and biased actions for the default noise and bias values set in `test_ours/py`. To change the noise and bias values, provide `--noise` and `--bias` as arguments with the training command.

We provide a pretrained model for $\tilde g$ in the `g_data/g_tilde/model_2objs_500`. To test StROL in the Robot environment, run
```bash
python3 test_ours.py --eval --boltzmann
```
This code uses a boltzmann rational model of the human to provide corrections to the robot. The simulated human, by default, chooses their actions from a binormal distribution of tasks. To use a uniform prior for the tasks, add the argument `--uniform` when running the script. The results for the runs for all approaces will be saved in `/results'.

### User Study
In our in-person user-study, the participants interact with a 7-DoF Franka Emika Panda robot arm to teach it 3 different tasks. The state and action space for one task is 3-dimensional, i.e. $x, u \in \mathbb{R}^3$, while for the other two tasks the state and action spaces are 6-dimensional ($x, u \in \mathbb{R}^6$). The robot is carrying a cup and its workspace consists of 2 objects (a pitcher and a plate). For the $1st$ task, the robot had access to three features, while for the $2nd$ and $3rd$ task the robot was given 4 features. For all tasks, $\hat g$ was trained with multinormal priors with number of possible tasks equal to the number of features.

The tasks, features for the tasks and the original learning rule $g$ of the robot for the user study are illustrated below.

#### Features
- Distance from the plate $d_{plate}$
- Distance from the pitcher $d_{pitcher}$
- Height from the table $h$
- Orientation of the end-effector $O_{ee}$

#### Tasks
- Task 1: The robot starts at a fixed position, holding a cup upright. The users taught the robot to move to the plate while avoiding the pitcher and keeping the cup close to the table.

- Task 2: The robot starts at a fixed position, holding the cup in a tilded position. The users taught the robot to carry the cup upright while moving to the plate, avoiding the pitcher and moving close to the table.

- Task 3: The robot starts in a similar pose as Task 2. The users taught the robot to carry the cup upright, while moving away from the plate, the pitcher and the table.

Note that Task 1 and Task 2 were incorporated in the prior, while Task 3 was a new task that was not included in the prior.

To implement StROL in a user study, move to the `user_study` folder using
``` bash
cd user_study
```

The pre-trained model of $\tilde g$ for Task 1 is saved in `g_data/model_t1`, and the pre-trained model for Task 2 and Task 3 is saved in `g_data/model_t2'.
To run tests on the robot, run the following command:
```bash
python3 user_study.py --eval --alg <algorithm> --task <task number> --env_dim <3/6> --n_features <3/4>
```
`--alg` defines the algorithm being used for the test - 'strol', 'oat' or 'mof', `--task` takes in the task number, i.e. 1, 2 or 3, `--env_dim` should be 3 for Task 1 and 6 otherwise and `--n_features` is 3 for Task 1 and 4 for the other tasks. If you want the robot to play the optimal robot trajectory for a given task use `--demo` argument when running the script.


## Results

## Computational Overheads
