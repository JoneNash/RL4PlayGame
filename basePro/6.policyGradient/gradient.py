#/bin/python
#coding=utf-8
import sys
sys.path.append("./secret")

from policy_value import *
import grid_mdp
import random
random.seed(0)
import numpy as np


def update_valuepolicy(valuepolicy, f, a, tvalue, alpha):
    pvalue        = valuepolicy.qfunc(f, a);
    error         = pvalue - tvalue; 
    fea           = valuepolicy.get_fea_vec(f, a);
    valuepolicy.theta -= alpha * error * fea;     

# policy 待更新的策略
# f      状态特征
# a      动作
# qvalue q值
# alpha  学习率
def update_softmaxpolicy(softmaxpolicy, f, a, qvalue, alpha):

    fea  = softmaxpolicy.get_fea_vec(f,a);
    prob = softmaxpolicy.pi(f);
    
    delta_logJ = fea;
    for i in xrange(len(softmaxpolicy.actions)):
        a1          = softmaxpolicy.actions[i];
        fea1        = softmaxpolicy.get_fea_vec(f,a1);
        delta_logJ -= fea1 * prob[i];
    delta_logJ     *= -1.0;

    softmaxpolicy.theta -= alpha * delta_logJ * qvalue; 

################ Different model free RL learning algorithms #####
def mc(grid,policy, num_iter1, alpha):
    actions = grid.actions;
    gamma   = grid.gamma;
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):

        f_sample = []
        a_sample = []
        r_sample = []   
        
        f = grid.start()
        t = False
        count = 0
        while False == t and count < 100:
            a = policy.take_action(f)
            t, f1, r  = grid.receive(a)
            f_sample.append(f)
            r_sample.append(r)
            a_sample.append(a)
            f = f1            
            count += 1


        g = 0.0
        for i in xrange(len(f_sample)-1, -1, -1):
            g *= gamma
            g += r_sample[i];
        
        for i in xrange(len(f_sample)):
            update_softmaxpolicy(policy, f_sample[i], a_sample[i], g, alpha)

            g -= r_sample[i];
            g /= gamma;
        

    return policy

def sarsa(grid, evaler, softmaxpolicy, valuepolicy, num_iter1, alpha):
    actions = grid.actions;
    gamma   = grid.gamma;
    y       = [];
    for i in xrange(len(valuepolicy.theta)):
        valuepolicy.theta[i]  = 0.1
    for i in xrange(len(softmaxpolicy.theta)): 
        softmaxpolicy.theta[i] = 0.0;
    

    for iter1 in xrange(num_iter1):
        y.append(evaler.eval(valuepolicy))
        f = grid.start();
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0

        while False == t and count < 100:
            t,f1,r      = grid.receive(a)
            a1          = softmaxpolicy.take_action(f1)
            update_valuepolicy(valuepolicy, f, a, r + gamma * valuepolicy.qfunc(f1, a1), alpha);
            update_softmaxpolicy(softmaxpolicy, f, a, valuepolicy.qfunc(f,a), alpha);

            f           = f1
            a           = a1
            count      += 1

    return softmaxpolicy, y;


