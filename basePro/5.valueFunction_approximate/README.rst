======================
价值函数近似
======================

在第三部分，我们采用价值表示策略；在第四部分，我们采用"价值-动作"表示策略，这两部分适合状态有限的情况。状态数量巨大时，就不再适合采用这种方式。
状态转移的可能性太多，会让整个模型变得非常复杂。


一种简化的方式是用特征去表示状态。到这里，增强学习与机器学习终于汇合了。
这里采用随机梯度下降。

需要特别注意的是，这里的error=上一轮更新的theta参数计算出来的状态价值 减去 上一轮的状态价值。
接下来就根据残差更新theta。循环往复直到收敛。


Modules
---------------
1. policy.py is the policy module

2. evaluate.py is the eval module

3. model_free.py contains different algorithm implements
 
4. experiment.py are the experiments scripts

 -You can use command "python experiment.py" to reproduce the experiments in the blog




