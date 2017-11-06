#coding=utf-8
#/bin/python
import numpy;
import random;

class Mdp:

    def __init__(self):

        #状态
        self.states         = [1,2,3,4,5,6,7,8] # 0 indicates end

        #
        self.terminal_states      = dict()
        self.terminal_states[6]   = 1
        self.terminal_states[7]   = 1
        self.terminal_states[8]   = 1

        #动作
        self.actions        = ['n','e','s','w']

        #奖励：R_s,a 表示状态 s 下采取动作 a 获得的奖励
        self.rewards        = dict();
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        #状态转移概率（字典）
        #也可以看做状态转移概率，P-s′_s,a 表示状态 s 下采取动作 a 之后转移到 s' 状态的概率
        #这里，转移概率都是100%
        self.t              = dict();
        self.t['1_s']       = 6
        self.t['1_e']       = 2
        self.t['2_w']       = 1
        self.t['2_e']       = 3
        self.t['3_s']       = 7
        self.t['3_w']       = 2
        self.t['3_e']       = 4
        self.t['4_w']       = 3
        self.t['4_e']       = 5
        self.t['5_s']       = 8 
        self.t['5_w']       = 4

        #衰减因子
        self.gamma          = 0.8

    #状态转移
    #返回一下个状态的信息（是否是最终状态，状态编号，奖励）
    def transform(self, state, action): ##return is_terminal,state, reward
        #当前状态为最终状态
        if state in self.terminal_states:
            return True, state, 0

        key = '%d_%s'%(state, action);
        #在状态转移字典中
        if key in self.t:
            next_state = self.t[key]; 
        #不在字典中
        else:
            next_state = state       
 
        is_terminal = False

        #下一个状态为最终状态
        if next_state in self.terminal_states:
            is_terminal = True
      
        #奖励值
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key];
           
        return is_terminal, next_state, r;
