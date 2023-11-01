import argparse
from g_func_exp.train_mpc import TRAIN
from g_func_exp.test_mpc import EVALUATE
import numpy as np


def main():

	p = argparse.ArgumentParser()
	p.add_argument('--train', action='store_true', default=False)
	p.add_argument('--eval', action='store_true', default=False)
	p.add_argument('--inv_g', action='store_true', default=False)
	p.add_argument('--boltzmann', action='store_true', default=False)
	p.add_argument('--uniform', action='store_true', default=False, help='if true, use uniform prior for sampling human goals')

	p.add_argument('--n_eval', type=int, default=100)
	p.add_argument('--lr', type=float, default = 1e-3)
	p.add_argument('--lr_gamma', type=float, default=0.5)
	p.add_argument('--n_train_steps', type=int, default=2000)
	p.add_argument('--n_features', type=int, default=3)
	p.add_argument('--hidden_size', type=int, default=256)
	p.add_argument('--env_dim', type=int, default=2)
	p.add_argument('--batch_size', type=int, default=100)
	p.add_argument('--noise', type=float, default=0.05)
	p.add_argument('--bias', type=float, default=0.0)

	args = p.parse_args()

	
	
	if args.train:
		TRAIN(args)

	if args.eval:
		EVALUATE(args)

if __name__=='__main__':
	main()

