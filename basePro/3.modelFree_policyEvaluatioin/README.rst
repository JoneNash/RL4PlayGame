======================
强化学习系列之三:模型无关的策略评价
======================

模型无关的策略评价，不知道马尔科夫决策过程转移概率和奖励函数。需要在实验环境中去探索。

模型无关的策略评价主要包括两种算法：蒙特卡洛算法，时差学习算法。

在蒙特卡洛算法中，每一个episode代表agent一个生命周期。在算法实现中，采用三个list分别存储当前状态s、即将采取的动作a、即将获得的奖励r。
多个episode组成三个二维list，这些可以看作是与环境交互得到的样本数据。
根据episode组成的集合，可以用来计算每个状态的价值。这一部分过程比较简单，基本就是求均值的过程。

时差学习方法中，方法略有不同。当前状态的价值至于当前状态采取动作的奖励以及下一个状态的价值有关。而且，下一个状态的价值从最新的状态价值表中获取。

相对于MC来说，TD只需要本次动作奖励、下一个状态（动作引起的状态）的价值两部分参与计算。非常适合难以获取完整"状态-动作-奖励"序列的场景，例如，飞行器控制。


