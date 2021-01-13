import numpy as np
import matplotlib.pyplot as plt


class MazeEnvironment(object):
    def __init__(self, maze_length):
        """Setup initial maze environment of maze_length length
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """

        # Initialize environment variables, updated in env_start()

        reward = None
        state = None
        termination = None
        self.reward_state_term = (reward, state, termination)

        # Create random maze
        self.maze_coords = np.zeros((1,2))
        self.start_loc = (0,0)
        self.maze = [self.start_loc]
        self.curr_point = self.start_loc

        # Start at 1 because initial point is defined above
        i = 1
        self.maze_complete = False
        while not self.maze_complete:

            action = np.random.choice([0,1,2,3],1)
            if action == 0:
                xstep = 0
                ystep = 1

            elif action == 1:
                xstep = 1
                ystep =0

            elif action == 2:
                xstep = 0
                ystep = -1

            elif action == 3:
                xstep = -1
                ystep = 0

            new_x = self.curr_point[0] + xstep
            new_y = self.curr_point[1] + ystep

            self.curr_point = (new_x,  new_y)
            if self.curr_point not in self.maze:
                self.maze.append(self.curr_point)
                self.maze_coords = np.append(self.maze_coords, np.array(
                    self.curr_point).reshape((1,2)), axis = 0)

                i += 1
                if i >= maze_length:
                    self.goal_loc = self.curr_point
                    self.maze_complete = True


    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts learning.

        Returns:
            The first state from the environment.
        """

        reward = 0
        # agent_loc holds the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = self.state(self.agent_loc)
        termination = False
        self.reward_state_term = (reward, state, termination)

        return self.reward_state_term[1]

    def env_step(self, state, action):
        """A step taken by the environment, used for simulation algorithms.

        Args:
            state: Current state
            action: The action taken by the agent


        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        x, y = self.maze[state]
        self.agent_loc = (x, y)

        # UP
        if action == 0:
            y += 1

        # Right
        elif action == 1:
            x += 1

        # Down
        elif action == 2:
            y -= 1

        # Left
        elif action == 3:
            x -= 1

        # Flag invalid move
        else:
            raise Exception(str(action) + " is an invalid move")

        # if the action takes the agent out-of-bounds
        # then the agent stays in the same state
        # otherwise, assign new position
        if (x, y) in self.maze:
            self.agent_loc = (x, y)

        # assign reward and termination status
        # reward defined as -1 per time step
        # to incentvize maze running speed
        reward = -1
        terminal = False

        # assign the reward and terminal variables
        if self.agent_loc == self.goal_loc:
            terminal = True
            reward = 0

        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term

    def env_end(self, reward):
        """Reset agent at end of episode"""
        self.agent_loc = self.start_loc

    def print_maze(self):
        fig, ax = plt.subplots()
        ax.scatter(self.maze_coords[:, 0], self.maze_coords[:,1])
        ax.scatter(self.start_loc[0], self.start_loc[1], c = "G", s = 150, marker = 'o')
        ax.scatter(self.goal_loc[0], self.goal_loc[1], c = "R", s = 150, marker = 's')
        plt.show()

    # helper method to get one dimensional state representation
    def state(self, loc):
        "Return the state encoding for maze location loc"
        return self.maze.index(loc)

    # helper method to get length of MazeEnvironment
    def __len__(self):
        return len(self.maze)
