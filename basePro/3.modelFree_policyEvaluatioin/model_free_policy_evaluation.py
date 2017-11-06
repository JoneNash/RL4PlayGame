#coding:utf-8
#!/bin/python
# import sys
# sys.path.append("./secret")
import grid_mdp
import random

grid     = grid_mdp.Grid_Mdp(); 
states   = grid.getStates();
actions  = grid.getActions(); 
gamma    = grid.getGamma();


'''
蒙特卡洛方法
输入——
gamma:衰减系数
state_sample:样本集合的状态集，二维数组
action_sample：样本集合的动作集，二维数组
reward_sample：样本集合的奖励集，二维数组
输出——
各个状态的状态价值
'''
def mc(gamma, state_sample, action_sample, reward_sample):   
    vfunc = dict();
    nfunc = dict();

    count=0;
    for s in states:
        #存储器S(s)和计数器N(n)
        vfunc[s] = 0.0
        nfunc[s] = 0.0 

    #每一个样本/序列参与一次计算
    #各个样本之间共享存储器和计数器
    for iter1 in xrange(len(state_sample)):
        G = 0.0
        #每个样本中的每个状态计算一次;倒序计算，得到第一个动作的价值
        for step in xrange(len(state_sample[iter1])-1,-1,-1):
            # print "step:",step
            count +=1;
            G *= gamma;
            G += reward_sample[iter1][step];

        #正序计算，根据第一个动作的价值，以及每次动作的奖励、衰减系数，反算后续动作的价值
        for step in xrange(len(state_sample[iter1])):
            s         = state_sample[iter1][step]
            vfunc[s] += G;
            nfunc[s] += 1.0;
            G        -= reward_sample[iter1][step]
            G        /= gamma;
    print "count:",count
    print "origin vfunc:",vfunc
    print "origin nfunc:", nfunc

    for s in states:
        if nfunc[s] > 0.000001:
            vfunc[s] /= nfunc[s]

    print "mc"
    print vfunc
    return vfunc


def td(alpha, gamma, state_sample, action_sample, reward_sample):
    #存储器，存储每个状态的价值
    vfunc = dict()
    for s in states:
        vfunc[s] = random.random()         
 
    for iter1 in xrange(len(state_sample)):
        for step in xrange(len(state_sample[iter1])):
            s = state_sample[iter1][step]
            r = reward_sample[iter1][step]

            #如果没有达到最后一个状态
            if len(state_sample[iter1]) - 1 > step:
                #下一个状态
                s1 = state_sample[iter1][step+1]
                #下一个状态的价值
                next_v = vfunc[s1]
            else:
                next_v = 0.0;

            # 这种方法会从第一个状态开始，一次更新各个状态的价值
            #这种时差学习算法对后续步骤不关心，我们称这种时差学习算法为TD(0)
            vfunc[s] = vfunc[s] +  alpha * (r + gamma * next_v - vfunc[s]);           


    print ""
    print "td"    
    print vfunc
    return vfunc


if __name__ == "__main__":
    #s,a,r 表示状态采取动作所获得的奖励组成的序列
    s, a, r = grid.gen_randompi_sample(5)
    print "s:",s
    print "a:",a
    print "r:",r

    s, a, r = grid.gen_randompi_sample(1000)

    vfunc=mc(0.5, s, a, r)
    print "vfunc:",vfunc
    # td(0.15, 0.5, s, a, r)
