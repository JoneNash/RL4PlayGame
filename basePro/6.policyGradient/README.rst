======================
策略梯度
======================

第五部分介绍的价值函数近似，用模型拟合价值函数。
接下来要说的梯度策略，用模型直接拟合策略。

针对离散的强化学习场景，由状态抽取状态特征向量s-hat，使用softmax策略计算f(s-hat,a)的概率。
针对连续的强化学习场景，由状态抽取状态特征向量s-hat，由于状态有无限多个，使用高斯策略计算f(s-hat,a)的概率。

策略梯度的第一个优势是其能够处理连续场景。价值函数近似就不适用了连续的强化学习场景。

eval.data是怎么来的？
model-based的方法计算出各个状态的价值，根据 " q(s,a)=Rs,a + 后续递减奖励之和 "计算。