# mcts_visualization
An implementation of visualizing Monte Carlo Tree Search 

Require: <code>pip install pygame</code>.

Run <code>run.py</code> to play.

在原作者基础上，整合了快手的[wu-uct](https://arxiv.org/pdf/1810.11755.pdf)并行MCTS算法，达到同快手论文基本一致的效果。

## Enviroment
It's a simplified version of an old Chinese chess.
<ul>
<li>When chess A jump over chess B, B will be removed.</li>
<li>The fewer chess left, the higher the score get.</li>
</ul>

## Algorithm
**Monte Carlo Tree Search (MCTS)** is a classic planning algorithm applied in Alphago and AlphaZero.
In this case, a pure MCTS (without value evaluation network) is used but still effective to solve the problem.

## Visualization logic
<ul>
<li>Child nodes inherit the color of the parent node.</li>
<li>Nodes at the same search depth are rendered in the same line.</li>
</ul>

## Video
<img src="images/mcts_restmin_visual.gif">


