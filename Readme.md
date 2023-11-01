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


## Training StROL
First, enter the environment in which you wish to run the simulation (Robot or CARLO) using `mkdir simulations/robot/` or `mkdir simulations/CARLO/`

## Training StROL
In StROL, we modify the learning dynamics of the system to incorporate a correction term in order to make learning from noisy and suboptimal actions more robust.

$$
\tilde g = g + \hat g
$$

For our implementation we formulate the correction term $\hat g$ as a fully connected neural network.

### Robot Environment
