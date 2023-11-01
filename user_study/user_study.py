import numpy as np
import pickle
import argparse
import  os, sys
from test import Task_1, Task_2, Task_3

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval', action='store_true', default=False)

    p.add_argument('--alg', type=str, required=True)
    
    

    p.add_argument('--task', type=int, default=1)
    p.add_argument('--user', type=str, default='test')
    p.add_argument('--demo', action='store_true', default=False)

    p.add_argument('--env_dim', type=int, default=3)
    p.add_argument('--n_features', type=int, default=3)
    p.add_argument('--hidden_size', type=int, default=256)

    args = p.parse_args()

    if not os.path.exists('data/user_{}/task_{}/{}/corrections'.format(args.user, args.task, args.alg)):
        os.makedirs('data/user_{}/task_{}/{}/corrections'.format(args.user, args.task, args.alg))

    if args.task == 1:
        run_study = Task_1(args)

    elif args.task == 2:
        run_study = Task_2(args)

    elif args.task == 3:
        run_study = Task_3(args)

    run_study.test()





if __name__=='__main__':
    main()