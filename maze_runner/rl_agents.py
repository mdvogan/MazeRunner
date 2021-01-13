import numpy as np
import matplotlib.pyplot as plt

class BaseAgent(object):
    def __init__(self, agent_info={}):
        """Setup for the agent"""

        # Set random number generator for reproducability
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Retrieve agent policy, discount facor, and learning rate parameters
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")

        self.nactions = self.policy.shape[1]

        # Initialize an array of zeros that will hold the values.
        # Note: policy is a State x Action dimensional matrix
        # with conditional probabilities, so the first dimension
        # is simply the number of states, in this case points in the maze
        self.values = np.zeros((self.policy.shape[0],))

    def argmax(self, values):
        """argmax with random tie-breaking
        Args:
            values (Numpy array): the array of values
        Returns:
            index (int): index of the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(values)):
            if values[i] > top:
                top = values[i]
                ties = []

            if values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def explore_maze(self, env, plot = False):
        """Explore maze according to current Agent policy
            Args: MazeEnvironment, plot command
            Returns: agent path and/or plot
        """

        state = env.env_start()
        path = np.array([[env.start_loc[0], env.start_loc[1]]])
        np.zeros((1,2))
        maze_complete = False

        while not maze_complete:

            #print("Start: {}, End: {}, Current Position: {}".format(env.start_loc, env.goal_loc, env.maze[state]))
            action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
            reward, state, terminal = env.env_step(state, action)
            loc = env.maze[state]
            path = np.append(path, np.array([[loc[0], loc[1]]]), axis = 0)

            if terminal:
                maze_complete = 1

        if plot == True:
            fig, ax = plt.subplots()
            ax.scatter(env.maze_coords[:, 0], env.maze_coords[:,1])
            ax.scatter(env.start_loc[0], env.start_loc[1], c = "G", s = 150, marker = 'o', label = "Start")
            ax.scatter(env.goal_loc[0], env.goal_loc[1], c = "R", s = 150, marker = 's', label = "End")
            ax.scatter(path[:, 0], path[:,1], c = "Y", label = "Optimal Policy")
            ax.set_title("Optimal MazeRunner Policy")
            ax.legend()
            plt.show()


        return path

class DPAgent(BaseAgent):

    def learn_value_iteration(self, env, theta):
        """Solves for optimal policy using dynamic programming value iteration
            finds optimal policy by bootstrapping state value function estimates
            and greedifying policy at each step. No sim. required
        Args: MazeEnvironment object, convergence epsilon (float)
        returns: optimal policy array and convergence information"""

        # Initialize V(s) arbitrarily except V(terminal) = 0
        self.values = np.zeros(len(env))

        # On each sweep, greedify w/r/t current action using bootstrapped V(s) estimates
        # until convergence is achieved and optimal policies identified
        optimal_policy = np.zeros(len(self.values), int)
        delta = 1
        iter = 0

        while delta >= theta:
            iter += 1
            delta = 0
            for s in range(len(self.values)):
                v_s = self.values[s]
                values = []
                for a in range(self.nactions):
                    reward, next_s, terminal = env.env_step(s, a)
                    update = reward + self.discount*self.values[next_s]
                    values.append(update)

                max_a = self.argmax(values)
                optimal_policy[s] = max_a
                self.values[s] = values[max_a]
                delta = max(delta, abs(v_s - self.values[s]))

            print("Iteration = {} | Delta = {}".format(iter, delta))

        # Overwrite with optimal policies
        self.policy = self.policy * 0
        for s in range(len(self.values)):
            self.policy[s,int(optimal_policy[s])] = 1

        return self.policy
