#!/bin/python
#coding=utf-8
import sys
sys.path.append("./secret")
import grid_mdp
import random
random.seed(0)
import matplotlib.pyplot as plt

grid     = grid_mdp.Grid_Mdp(); 
states   = grid.getStates();
actions  = grid.getActions(); 
gamma    = grid.getGamma();

###############   Compute the gaps between current q and the best q ######
best = dict();
def read_best():
    f = open("best_qfunc")
    for line in f:
        line = line.strip()
        if len(line) == 0:  continue
        eles              = line.split(":")
        best[eles[0]] = float(eles[1])

def compute_error(qfunc):
    sum1 = 0.0
    for key in qfunc:
        error = qfunc[key] - best[key]
        sum1 += error * error
    return sum1



##############   epsilon greedy policy #####
#epsilon贪婪策略
def epsilon_greedy(qfunc, state, epsilon):
    ## max q action
    #最优动作的下标
    amax    = 0
    key     = "%d_%s"%(state, actions[0])
    qmax    = qfunc[key]
    # 遍历动作集合，选择一个q值最大的动作
    for i in xrange(len(actions)):
        key = "%d_%s"%(state, actions[i])
        q   = qfunc[key]
        if qmax < q:
            qmax  = q;
            amax  = i; 
    
    ##probability
    #给不同的动作分配不同的概率，其中最优动作的概率是： 1-epsilon + epsilon / len(actions)
    pro = [0.0 for i in xrange(len(actions))]
    pro[amax] += 1- epsilon
    for i in xrange(len(actions)):
        pro[i] += epsilon / len(actions)

    ##choose
    #通过累加之后的概率与随机值比较，确定最终的动作
    r = random.random()
    s = 0.0
    for i in xrange(len(actions)):
        s += pro[i]
        if s >= r: return actions[i]
    return actions[len(actions)-1]

################ Different model free RL learning algorithms #####

#蒙特卡洛方法：实验样本数，贪婪系数
def mc(num_iter1, epsilon):
    x = []
    y = []
    n     = dict();
    qfunc = dict();
    for s in states:
        for a in actions:
            qfunc["%d_%s"%(s,a)] = 0.0
            n["%d_%s"%(s,a)] = 0.001

    #每一轮实验，都会更新qfunc
    for iter1 in xrange(num_iter1):
        x.append(iter1);
        #计算每一轮的q与理想q的方差
        y.append(compute_error(qfunc))

        s_sample = []
        a_sample = []
        r_sample = []   
        
        #随机选择一个状态
        s = states[int(random.random() * len(states))]

        t = False
        count = 0
        #状态转移，最多转移100次，最终产生单次实验样本
        while False == t and count < 100:
            #确定动作
            a = epsilon_greedy(qfunc, s, epsilon)
            #状态s采取a动作之后，获取（s1是否是最终状态，新状态，奖励）
            t, s1, r  = grid.transform(s,a)
            s_sample.append(s)
            r_sample.append(r)
            a_sample.append(a)
            s = s1            
            count += 1

        g = 0.0
        #计算第一个状态的递减奖励
        for i in xrange(len(s_sample)-1, -1, -1):
            g *= gamma
            g += r_sample[i];
                
        #利用单次实验，计算各个状态的qfunc
        for i in xrange(len(s_sample)):
            key = "%d_%s"%(s_sample[i], a_sample[i])
            n[key]      += 1.0;
            #n次的均值
            qfunc[key]   = (qfunc[key] * (n[key]-1) + g) / n[key]            
 
            g -= r_sample[i];
            g /= gamma;

    plt.plot(x,y,"-",label="mc epsilon=%2.1f"%(epsilon));
    return qfunc


#时差学习算法：实验样本数，学习率，贪婪系数
def sarsa(num_iter1, alpha, epsilon):
    x = []
    y = []
    qfunc = dict();
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0

    for iter1 in xrange(num_iter1):

        x.append(iter1)
        y.append(compute_error(qfunc))

        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0
        while False == t and count < 100:
            key         = "%d_%s"%(s,a)
            #grid这里像一个黑盒子
            t,s1,r      = grid.transform( s,a)
            a1          = epsilon_greedy(qfunc, s1, epsilon)
            key1        = "%d_%s"%(s1,a1)

            #使用下一状态q值更新当前q值，这里与mc方法存在非常大的不同
            qfunc[key]  = qfunc[key] + alpha * ( \
                          r + gamma * qfunc[key1] - qfunc[key])
            s           = s1
            a           = a1
            count      += 1

    plt.plot(x,y,"--",label="sarsa alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
    return qfunc;

def qlearning(num_iter1, alpha, epsilon):
    x = []
    y = []
    qfunc = dict()
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0


    for iter1 in xrange(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))

        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0
        while False == t and count < 100:
            key         = "%d_%s"%(s,a)
            t,s1,r      = grid.transform(s,a)

            key1 = ""
            qmax = -1.0
            for a1 in actions:
                if qmax < qfunc["%d_%s"%(s1,a1)]:
                    qmax = qfunc["%d_%s"%(s1,a1)]
                    key1 = "%d_%s"%(s1,a1)
            qfunc[key]  = qfunc[key] + alpha * ( \
                          r + gamma * qfunc[key1] - qfunc[key])

            s           = s1
            a           = epsilon_greedy(qfunc, s1, epsilon)
            count      += 1   

    plt.plot(x,y,"-.,",label="q alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
    return qfunc

