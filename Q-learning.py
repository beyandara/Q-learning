import random as rnd

class Robot:
    def __init__(self, alpha = 0.3, gamma = 0.5, epsilon = 0.8):
        # Initialize the robot with reward matrix, action-to-moves dictionary and parametres initializing the class 
        self.reward_matrix = [
            [-1000, -800, -800, 150, 150, -1000],
            [-1000, -1000, 150, -800, 150, 150],
            [-800, 150, 150, -800, 150, -800],
            [-800, 150, 150, 150, 150, 150],
            [-800, 150, -800, 150, -800, 150],
            [10000, -1000, -1000, -1000, -1000, -1000]
        ]
        # The action-to-moves dictionary contains possible choices and their changes in x and y coordinates
        self.action_dict = {0: [-1, 6], 1: [1, 1], 2: [1, 6], 3: [-1, 1]}
        # parametres for the Q-learnign algorithm
        self.alpha = alpha  # learning speed
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration probability

    # method to choose the next state for monte-carlo-simulation
    def get_next_state_mc(self):
        return rnd.randint(0, 3)  # random choice of action
    
     # Method to pick the next state using greedy-epsilon
    def get_next_state_eg(self, x, y):
        if rnd.random() < self.epsilon:
            return rnd.randint(0, 3)  # get a random action on epsilon probability
        return self.q_matrix[(x, y)].index(max(self.q_matrix[(x, y)]))  # chooses best action based on q_matrix 
                                                                        # on epsilon 1 - probability

    # method to calculate reward for an action
    def get_reward(self, x, y, action):
        # retrieves operator and move which tell us which way to agent will be moving
        operator, move = self.action_dict[action][0], self.action_dict[action][1]
        original_x = x
        original_y = y
        if move == 6:
            x += operator  # moving up or down
        else:
            y += operator  # moving left or right

        # check to see if it's a legal move
        if x in (-1, 6) or y in (-1, 6):
            return -9999999, original_x, original_y  # returns a low number of -9999999 and the unchanged x and y
        return self.reward_matrix[x][y], x, y  # returns reward and the new state's x and y


    # Method for monte-carlo-exploration
    def monte_carlo_exploration(self, stimulation=100):
        highest_total_reward = -9999999

        for i in range(stimulation):
            current_pos = (0, 3)  # starting from the start position
            current_route = [current_pos]
            total_reward = 0

            goal_found = False
            while not goal_found:
                action = self.get_next_state_mc()  # chooses a random action
                try:
                    current_reward, x, y = self.get_reward(current_pos[0], current_pos[1], action)
                    total_reward += current_reward
                    current_pos = (x, y)  # updates "current position"
                    current_route.append(current_pos)  # adds new (current) position to the current path/route
                    goal_found = current_pos == (5, 0)  # check to see if the current position is the destination
                except:
                    continue

            if total_reward > highest_total_reward:
                best_route = current_route  # updates best_route to the best route
                highest_total_reward = total_reward  # updates the highest_reward

        return best_route, highest_total_reward  # returns both the best route and highest total reward

    # Q-learning algorithm
    def q_learning(self, epoch):
        self.position = [(i, j) for i in range(6) for j in range(6)] # position is a two dimensional list of all possible states
        self.q_matrix = {j: [0] * 4 for j in self.position} # Initializing Q-matrix with zeros for every state and every action 
        self.q_position = [[0] * 4 for _ in self.position]  # q_position is creating a 2 dimensional list with 36x4 nested lists
                                                            # that at the end of this method will represent each states 
                                                            # "next-possible-state"    

        for _ in range(epoch):
            x = rnd.randint(0, 5)  # each episode, start from a random posistion 
            y = rnd.randint(0, 5)
            current_pos = (x, y)

            goal_found = False
            while not goal_found:
                action = self.get_next_state_eg(x, y)  # Pick an action based on epsilon greedy
                current_reward, x, y = self.get_reward(current_pos[0], current_pos[1], action)
                if current_reward > -9999999:
                    # Update the Q-matrix for the current state and the calculated Q_value 
                    self.q_matrix[current_pos][action] = round(
                        (1 - self.alpha) * self.q_matrix[current_pos][action] + self.alpha * (
                                    current_reward + self.gamma * max(self.q_matrix[(x, y)])), 0)
                    self.q_position[current_pos[0] * 6 + current_pos[1]][action] = (x, y)  # updates position on the q_position list

                    current_pos = (x, y)  # updates new current position
                    goal_found = current_pos == (5, 0)  # checks if at destination
                else:
                    if self.q_matrix[current_pos][action] != -9999999:
                        self.q_matrix[current_pos][action] = -9999999  # give illegal action a high punishment

        return self.q_matrix  # returns updated Q matrix after epoch episodes

    # Method to find the best way from start_position to goal by using  Q-values in the q_matrix
    def greedy_path(self, start_node, end_node=(5, 0)):
        action_graph = self.make_action_position_graph()  # create a graph that holds the best actions based on the Q-values
        current_node = start_node
        goal_found = False
        path = [current_node]

        while not goal_found:
            if current_node == end_node:
                goal_found = True  # if goal found, exit the search
                break
            max_inner_key = self.find_max_key(action_graph[current_node])  # find the action with the highest Q-value

            current_node = max_inner_key  # update current_node to the best next-state
            path.append(current_node)  # add node to path

        return path  # Returns calculated path

    # Method to create a graph that contains the best possible actions + moves based on the Q-values
    def make_action_position_graph(self):
        action_graph = {}
        for position, q_values in self.q_matrix.items():
            new_q_values = {}
            for i, q_value in enumerate(q_values):
                if isinstance(q_value, float):
                    new_q_values[self.q_position[position[0] * 6 + position[1]][i]] = q_value
            action_graph[position] = new_q_values

        return action_graph  # returns graph that will assist the greedy path algorithm

    # Method to find the key value that holds the highest value in the "inner dictionary, meaning
    # the possible actions value
    def find_max_key(self, inner_dict):
        return max(inner_dict, key=inner_dict.get)

if __name__ == "__main__":
    """Here are five different tests, starting with the first three with 100 episodes. First test has standard 
        epsilon value of 0.8. Second is with a high value of epsilon = 0.9, and third is with a very low 
        epsilon = 0.1. The last two have epsilon value = 0.8 and an episode run of 50 and 25.
        
        As the tests don't all work optimally, I have commented out the last 4. Uncomment these 
        at a time and comment the tests not running."""
    
    start_pos = (0, 3)
    goel_pos = (5, 0)
    
    # Test 1: Standard parametres and epsilon-greedy policy
    test1 = Robot()
    q_matrix1 = test1.q_learning(100)  # 100 episodes
    path1 = test1.greedy_path(start_pos, goel_pos)
    mc = test1.monte_carlo_exploration()

    print("Test 1 - epsilon-greedy policy:")
    print("Optimal route:", path1, "\n")
    
    # print(mc, "\n")
    # print("Q_matrix:")
    # for position, q_values in q_matrix1.items():
    #     print(position, q_values)
    

    ###### Test 2: Standard parametres and epsilon-greedy = 1 ######
    """ High epsilon-greedy factor 
    greedy_path = 
    [(0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (3, 1), (4, 1), (4, 0), (5, 0)]
    ➔ Took significantly longer than test #1 due to the use of mostly random actions."""

    # test2 = Robot(epsilon=1)
    # q_matrix2 = test2.q_learning(100)  # 100 episodes
    # path2 = test2.greedy_path(start_pos, goel_pos)
    # mc = test2.monte_carlo_exploration()

    # print("\nTest 2 - epsilon-greedy policy:")
    # print("Optimal route:", path2)

    # print(mc, "\n")
    # print("Q_matrix:")
    # for position, q_values in q_matrix2.items():
    #     print(position, q_values)
    
 
    ###### Test 3: Standard parametres and epsilon-greedy = 0.1 ###### 
    """
    Low epsilon-greedy factor
    ➔ The program used calculated actions 90% of the time and random 
    actions 10% of the time. The program did not reach a result as
    the current_pos alternated between (0, 4) and (1, 4). These
    positions had the highest Q-value for each other, causing the
    Q-matrix to stop updating further."""
    # test3 = Robot(epsilon=0.1)
    # q_matrix3 = test3.q_learning(100) # 100 episodes
    # path3 = test3.greedy_path(start_pos, goel_pos)
    # mc = test3.monte_carlo_exploration()

    # print("\nTest 3 - epsilon-greedy policy:")
    # print("Optimal route:", path3)

    # print(mc, "\n")
    # print("Q_matrix:")
    # for position, q_values in q_matrix3.items():
    #     print(position, q_values)


    ##########################################################################
    ############### RE-TESTING the above with 50 episodes ####################
    ########################################################################## 
    
    ###### Test 4: Standard parametres and epsilon-greedy policy ###### 
    """Standard epsilon greedy
        greedy_path = 
        [(0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (3, 1), (4, 1), (4, 0), (5, 0)]

        The program arrived at the solution above in approximately 80%
        of the test runs but went into an infinite loop in the
        remaining 20%. The problem occurred in the greedy path 
        and was the result of insufficient exploration of the terrain."""
    # test4 = Robot()
    # q_matrix4 = test4.q_learning(50)  # 50 episodes
    # path4 = test4.greedy_path(start_pos, goel_pos)
    # mc = test4.monte_carlo_exploration()

    # print("Test 4 - Standard parametres and epsilon-greedy policy:")
    # print("Optimal route:", path4)

    # print(mc, "\n")
    # print("Q-matrix:")
    # for position, q_values in q_matrix4.items():
    #     print(position, q_values)

    ###########################################################################
    ################ RE-TESTING the above with 20 episodes ####################
    ########################################################################### 
    
    ###### Test 5: Standard parametres and epsilon-greedy policy ###### 
    """Standard epsilon greedy factor:
        Due to the limited number of episodes, the Q-matrix was 
        incomplete, and the program did not find an optimal path 
        despite running it more than 20 times. """
    # test5 = Robot()
    # q_matrix5 = test5.q_learning(25)  # 25 episodes
    # path5 = test5.greedy_path(start_pos, goel_pos)
    # mc = test5.monte_carlo_exploration()
    # print("Test 5 - Standard hyperparametere and epsilon-greedy policy:")
    # print("Optimal rute:", path5)
    # print("Q-matrix:")
    # for position, q_values in q_matrix5.items():
    #     print(position, q_values)
    
    """The tests that did not work optimally were run with 200 epochs, 
    up to 20 times. Even with this, it was not possible to achieve a 
    satisfactory result."""