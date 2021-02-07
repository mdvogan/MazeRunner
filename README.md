Tabular Reinforcement Learning for Maze Navigation

This is a practice project to implement tabular reinforcement
learning algorithms from scratch to navigate random mazes

Algorithms considered will include:

1) Dynamic programming value Iteration
2) Monte Carlo (on and off policy)
3) SARSA/Expected SARSA
4) Q-Learning

rl_agents.py creates a random maze environment and uses agents to learn optimal
exploration policies. The code runs fast on mazes up to 100 spaces large

Note: be careful with the MonteCarlo agent. Sometimes, because it must
wait for the end of the episode to learn, it can get stuck for a long
time which creates a large episode array that can slow things down
