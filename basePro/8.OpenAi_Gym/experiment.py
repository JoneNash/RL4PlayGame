#!/bin/python

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import random
random.seed(10)
from algorithms import *
from policy     import *
from mdp        import *

if __name__ == "__main__":
    cartpole =gym.make('CartPole-v0')
    policy   = Policy(cartpole, epsilon = 0.01) 
    policy   = sarsa(cartpole, policy,\
                   num_iter1 = 1000, \
                   alpha = 0.1, \
                   gamma = 0.9);


    for iter1 in xrange(20):
        s_f = cartpole.reset()
        for iter2 in range(2000):
            cartpole.render()
            a = policy.greedy(s_f)
            s_f, r, t, i = cartpole.step(a)
            if t:
                print "{} Episode finished after {} timesteps".format(iter1,iter2+1)
                break


