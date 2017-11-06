#coding=utf-8
#/bin/python
import numpy;
import random as ran;
ran.seed(0)
from mdp import *;

#随机决策
def random_pi():
    actions = ['n','w','e','s']
    r       = int(ran.random() * 4)
    return actions[r]

#计算状态价值
def compute_random_pi_state_value():
    value = [ 0.0 for r in xrange(9)]
    num   = 1000000

    for k in xrange(1,num):

        for i in xrange(1,6):
            mdp = Mdp();
            s   = i;
            is_terminal = False
            gamma = 1.0
            v     = 0.0

            while False == is_terminal:
                #随机选择一个动作
                a                 = random_pi()
                #根据当前状态、采取的动作，确定下一个状态及价值
                is_terminal, s, r = mdp.transform(s, a)
                #价值累加，考虑衰减系数
                v                += gamma * r;
                gamma            *= 0.5
  
            #k轮计算之后的均值（实际上，做多轮实验仅仅起到防止抖动引起的偏差，多轮实验相互不影响）
            value[i] = (value[i] * (k-1) + v) / k

        if k % 10000 == 0:
            print k,"------",value

    print value

compute_random_pi_state_value()
        


