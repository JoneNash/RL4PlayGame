#coding:utf-8
#/bin/python
import numpy;
import random;
from grid_mdp import Grid_Mdp


class Policy_Value:
    def __init__(self, grid_mdp):
        self.v  = [ 0.0 for i in xrange(len(grid_mdp.states) + 1)]
        
        self.pi = dict()
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states: continue
            self.pi[state] = grid_mdp.actions[ 0 ]
    
    #策略改进
    def policy_improve(self, grid_mdp):
   
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states: continue

            a1      = grid_mdp.actions[0]
            t, s, r = grid_mdp.transform( state, a1 )
            v1      = r + grid_mdp.gamma * self.v[s]    

            for action in grid_mdp.actions:
                t, s, r = grid_mdp.transform( state, action )
                if v1 < r + grid_mdp.gamma * self.v[s]:  
                    a1 = action
                    v1 = r + grid_mdp.gamma * self.v[s]                

            self.pi[state] = a1


    #策略评估
    def policy_evaluate(self, grid_mdp):
        # print("policy_evaluate :")
        for i in xrange(1000):

            delta = 0.0
            for state in grid_mdp.states:
                if state in grid_mdp.terminal_states: continue
                #使用每个状态上轮策略改进确定的动作
                action        = self.pi[state]
                #t：是否达到最终状态
                #s：state采取action之后的状态
                #r: state采取action之后获得奖励r
                t, s, r       = grid_mdp.transform(state, action)

                #计算新的状态价值:奖励+衰减系数*新状态当前的价值
                new_v         = r + grid_mdp.gamma * self.v[s]

                #是否停止更新
                delta        += abs(self.v[state] - new_v)

                self.v[state] = new_v


            if delta < 1e-6:
                # print("stop policy_evaluate in step ",i)
                break;

    #策略迭代
    def policy_iterate(self, grid_mdp):
        for i in xrange(100):
            self.policy_evaluate(grid_mdp);
            self.policy_improve(grid_mdp);


if __name__ == "__main__":
        grid_mdp     = Grid_Mdp()
        policy_value = Policy_Value(grid_mdp)
        print("policy_value_start :",policy_value.v,policy_value.pi)

        policy_value.policy_iterate(grid_mdp)

        print("policy_value_end :", policy_value.v, policy_value.pi)
        print "value:"
        for i in xrange(1,6):
            print "%d:%f\t"%(i,policy_value.v[i]),
        print ""
 
        print "policy:"
        for i in xrange(1,6):
            print "%d->%s\t"%(i,policy_value.pi[i]),
        print ""
