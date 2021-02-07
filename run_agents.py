import numpy as np
import matplotlib.pyplot as plt

import maze_runner as mr


# Initialize MazeEnvironment
env = mr.MazeEnvironment(25)
#env.print_maze()

# Initialize dynamic programming agent
# Create uniform random policy
rand_uni_policy = np.ones((len(env), 4)) * (1/4)
agent_info = {"policy": rand_uni_policy, "discount": 1, "step_size": 0.5, "epsilon": 0.01}
#dp_agent = mr.DPAgent(agent_info)

# Learn optimal policy through dynamic programming value iteration algorithm
#dp_agent.learn_policy(env, theta = 0.01)

# Explore maze using current learned policy
#dp_agent.explore_maze(env, plot = True)
sarsa_agent = mr.SARSAAgent(agent_info)
sarsa_agent.learn_policy(env, expected = True, n_episodes = 100)
sarsa_agent.explore_maze(env, plot = True)

q_agent = mr.QAgent(agent_info)
q_agent.learn_policy(env, n_episodes = 100)
q_agent.explore_maze(env, plot = True)

mc_agent = mr.MCAgent(agent_info)
mc_agent.learn_policy(env, n_episodes = 100)
mc_agent.explore_maze(env, plot = True)
