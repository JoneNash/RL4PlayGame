#!/usr/bin/env python
#coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
@version: ??
@author: leidelong
@license: Apache Licence
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/9/23 下午10:46
"""


import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break