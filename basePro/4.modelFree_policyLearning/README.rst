======================
模型无关的策略学习
======================


所谓模型无关的策略学习，指的是如何对环境进行探索。
在第三部分，当前状态采取动作时，采用的是随机策略，这种方式未免过于低效。一种选择动作的方式是贪婪法，即，选择接下来价值最大的状态，这种方法很容易进入局部最优。
e-贪婪策略算法则是在探索和利用上做平衡，与E&E具有相同的思想。

在之前，mc采用状态价值作为表示策略，在这里，将采用"状态-动作"的价值表示策略。
为什么采用"状态-动作"价值，不再采用状态价值？
因为在模型无关的场景中，是无法知道s采取a动作转移到s'的概率的。
在前面的试验中，其实有一个前提条件，s采取a动作转移到s'的概率是0、1的一种。
在现实当中，这种转移概率是未知的。


SARSA算法其实是状态-动作价值版本的时差学习 (Temporal Difference, TD) 算法。SARSA算法在计算下一个"状态-动作"也采用e-贪婪策略算法。

Q-learning与SARSA相似，不同点在于，计算下一个"状态-动作"时采用贪婪算法。

从实践来看，Q-learning收敛最快，收敛到最优策略。个人认为，是因为其采用前一个状态做探索、后者一个状态给精确值，是的每个"状态-动作"的价值更加准确。
MC Control 和 SARSA 则只能收敛到 e-贪婪策略。由于SARSA在后一个状态的状态-动作价值计算时采用e-贪婪策略，相当于引入了噪声，所以，不会收敛到最优策略。


Modules
---------------

1. model_free.py contains different algorithm implements
 
2. experiment_* are the experiments scripts

 -experiment_comprehensive.py is for the comprehensive experiment (chapter 5.3 in the blog)

 -experiment_epsilon_greedy.py is for the epsilon greedy experiment (chapter 5.2 in the blog)

 -experiment_near_optimal.py (chapter 2 in the blog)

 -experiment_variance.py is for the stability experiment (chapter 5.1 in the blog)




