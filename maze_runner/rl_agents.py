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
        self.epsilon = agent_info.get("epsilon", 0.1)

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


    def update_e_greedy(self, state):
        """epsilon greedy action selection
            Choose greedy action w/ prob (1-e + (e/#actions )),
            non greedy actions prob e/#actions
        Args:
            state: current state"""

        greedy_action = self.argmax(self.action_values[state, :])
        self.policy[state, :] = self.epsilon/self.nactions
        self.policy[state, greedy_action] = 1 - self.epsilon + (
            self.epsilon/self.nactions)

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

    def learn_policy(self, env, theta):
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

class MCAgent(BaseAgent):

    def learn_policy(self, env, n_episodes = 1000, offpolicy = False):
        """Solves for optimal policy using Monte Carlo simulation
        Args: MazeEnvironment object, number of episodes, on/off policy
        returns: optimal policy array and convergence information"""

        # Reset policy to rand uni just in case another learning algorithm
        # was already run
        self.policy = np.ones((len(self.values), self.nactions)) * (1/self.nactions)

        # Initialize action value and returns matrix
        self.action_values = np.ones((len(self.values), self.nactions))

        episode_lengths = []
        returns = {}
        for s in range(len(self.values)):
            for a in range(self.nactions):
                returns[(s, a)] = [0,0]

        for episode in range(n_episodes):
            #Generate episode according to current policy
            #Initialize starting point
            termination = False
            state = env.env_start()
            action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])

            #Episode sequence stored as reward, state, action
            episode_sequence = []
            while termination == False:
                reward, next_state, termination = env.env_step(state, action)
                episode_sequence.append((reward, state, action))

                next_action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[next_state])
                state =  next_state
                action = next_action

            print("Episode {} complete, length {}".format(episode, len(episode_sequence)))

            # Work backwards and calculate action values using simulated returns
            episode_lengths.append(len(episode_sequence))
            total_rewards = 0
            for step in range(len(episode_sequence) - 1, -1, -1):
                reward, state, action = episode_sequence[step]
                total_rewards = self.discount*total_rewards + reward

                # Update action value return averages
                returns[(state, action)][0] += 1
                n = returns[(state, action)][0]
                returns[(state, action)][1] = (total_rewards/n) + returns[(state, action)][1] * ((n-1)/n)

                self.action_values[state, action] = returns[(state, action)][1]

                self.update_e_greedy(state)

        return self.policy

class SARSAAgent(BaseAgent):

    def learn_policy(self, env, expected = False, n_episodes = 1000):
        """Solves for optimal policy using SARSA control algorithm
        Args: MazeEnvironment object, number of episodes, on/off policy
            expected: Flag for expected SARSA algorithm
        returns: optimal policy array and convergence information"""

        # Reset policy to rand uni just in case another learning algorithm
        # was already run
        self.policy = np.ones((len(self.values), self.nactions)) * (1/self.nactions)

        # Initialize action value and returns matrix
        self.action_values = np.ones((len(self.values), self.nactions))

        episode_lengths = []

        for episode in range(n_episodes):
            #Generate episode according to current policy
            #Initialize starting point
            termination = False
            state = env.env_start()
            action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])

            #Episode sequence stored as reward, state, action
            step_counter = 0
            while termination == False:
                step_counter += 1
                reward, next_state, termination = env.env_step(state, action)
                next_action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[next_state])

                if expected == False:
                    self.action_values[state, action] = self.action_values[state, action] + self.step_size*(
                        reward + self.discount*self.action_values[next_state, next_action] - self.action_values[state, action])
                else:
                    next_expected_value = np.dot(self.action_values[next_state, :], self.policy[next_state, :])
                    self.action_values[state, action] = self.action_values[state, action] + self.step_size*(
                        reward + self.discount*next_expected_value - self.action_values[state, action])

                self.update_e_greedy(state)

                state =  next_state
                action = next_action

            episode_lengths.append(step_counter)

        return self.policy

class QAgent(BaseAgent):

    def learn_policy(self, env, n_episodes = 1000):
        """Solves for optimal policy using q-learning control algorithm
        Args: MazeEnvironment object, number of episodes, on/off policy
            expected: Flag for expected SARSA algorithm
        returns: optimal policy array and convergence information"""

        # Reset policy to rand uni just in case another learning algorithm
        # was already run
        self.policy = np.ones((len(self.values), self.nactions)) * (1/self.nactions)

        # Initialize action value and returns matrix
        self.action_values = np.ones((len(self.values), self.nactions))

        episode_lengths = []

        for episode in range(n_episodes):
            #Generate episode according to current policy
            #Initialize starting point
            termination = False
            state = env.env_start()
            action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])

            #Episode sequence stored as reward, state, action
            step_counter = 0
            while termination == False:
                step_counter += 1
                reward, next_state, termination = env.env_step(state, action)

                next_action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[next_state])
                best_action = self.argmax(self.action_values[next_state, :])

                self.action_values[state, action] = self.action_values[state, action] + self.step_size*(
                    reward + self.discount*self.action_values[next_state, best_action] - self.action_values[state, action])

                self.update_e_greedy(state)

                state =  next_state
                action = next_action

            episode_lengths.append(step_counter)

        return self.policy
