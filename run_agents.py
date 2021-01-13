import numpy as np
import matplotlib.pyplot as plt

import maze_runner as mr


# Initialize MazeEnvironment
env = mr.MazeEnvironment(250)
#env.print_maze()

# Initialize dynamic programming agent
# Create uniform random policy
rand_uni_policy = np.ones((len(env), 4)) * (1/4)
agent_info = {"policy": rand_uni_policy, "discount": 1, "step_size": 1.5}
dpagent = mr.DPAgent(agent_info)

# Learn optimal policy through dynamic programming value iteration algorithm
dpagent.learn_value_iteration(env, 0.01)

# Explore maze using current learned policy
dpagent.explore_maze(env, plot = True)
