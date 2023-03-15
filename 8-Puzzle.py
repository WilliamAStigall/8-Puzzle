import random
import copy
import heapq
import time
#If I was to use path+current_state I would be able to generate a sequence of 3x3 states
#Technically my solution is not a sequence of moves, however each iteration a new move has been shown, so I kind of do it
# Reformatted using PEP Style Guides
##I will run each algorithm 10 times, which algorithm is selected to sort the state will be selected randomly
##This allows for me to not have to add input while selecting the
class RandomSelector:
    ##Initialize start_time variable to calculate runtime of the program
    start_time = time.time()

    # We use this to randomly select a searching algorithm on run
    def random_selector(self):
        ##randomly select a number 1 through 7 and print which algorithm we are running
        random_number = random.randint(1, 7)
        ##We have two different DFS Functions DFS is a nonworking throwaway function
        if random_number == 1:
            print("Implementing DFS")
        elif random_number == 2:
            print("Implementing BFS")
        elif random_number == 3:
            print("Implementing UCS")
        elif random_number == 4:
            print("Implementing A*")
        elif random_number == 5:
            print("Implementing Heuristics used in UCS")
        elif random_number == 6:
            print("Implementing Heuristics used in A*")
        elif random_number == 7:
            print("Implementing IDS")
        return random_number


##State class used to generate our random States
class State:
    # #randomize our state, with '-' for the emtpy space, this does not work for every algorithm that we use #I
    # probably should have figured out a way to make it blank #During UCS and A* algorithm's using the state
    # randomized from this function will result in a type error because the priority Queue #Cannot compare '-' to the
    # other integer values in the list, therefore we needed another working state randomizer for just integers with 0
    # representing our empty space
    def state_randomizer(self):
        element_list = [i for i in range(1, 9)] + ['-']
        random.shuffle(element_list)
        grid = [element_list[i:i + 3] for i in range(0, 9, 3)]
        return grid

    ##Create our goal grid
    def goal_state(self):
        goal_grid = [[1, 2, 3], [8, '-', 4], [7, 6, 5]]
        return goal_grid

    ##Initialize the same thing however to work with only integers
    def state_randomizer_int(self):
        element_list = [i for i in range(9)]
        random.shuffle(element_list)
        grid_int = [element_list[i:i + 3] for i in range(0, 9, 3)]
        return grid_int

    def goal_state_int(self):
        goal_grid_int = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
        return goal_grid_int


##in hindsight,this should have been implemented iterativley however, here we did it using recursion
##Recursive Method to find the goal State given the random State
class DepthFirstSearch:
    ##Initializes the class with our random state goal state and current state
    ##we get both random state and current state from the same randomized state
    def __init__(self, random_state, goal_state, current_state):
        self.random_state = random_state
        self.goal_state = goal_state
        self.current_state = current_state
    # #Check if the randomState is = to the goal_state, this is called each new state so we can be alerted when the
    # goal state is reached
    def check_goal(self, goal_state, current_state, num_nodes, random_state):
        current_state_str = "\n".join([" ".join([str(x) for x in row]) for row in current_state])
        ##If the current state is = to the goal State return true
        ##Checks if all the indexes are matched between the current State and the goal_state
        if all([current_state[i][j] == goal_state[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(num_nodes))
            print("Goal State: \n" + current_state_str + "\n")
            return True
        else:
            print("Current state:\n" + current_state_str+ "\nNodes visited so far:" + str(num_nodes) + "\n")
            return False
    ##Function to find the location visualized as the '-' String
    def get_empty_location(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == "-":
                    return (i, j)
    ##Function to get the possible moves that can be achieved by the empty location
    ##it appends each possible move in four directions to the list[] moves
    def get_possible_moves(self, state):
        moves = []
        i, j = self.get_empty_location(state)
        if i > 0:
            moves.append((i - 1, j))
            # Check if the empty cell can be moved down, add the new location to the moves list if possible
        if i < 2:
            moves.append((i + 1, j))
            # Check if the empty cell can be moved left, add the new location to the moves list if possible
        if j > 0:
            moves.append((i, j - 1))
            # Check if the empty cell can be moved right, add the new location to the moves list if possible
        if j < 2:
            moves.append((i, j + 1))
            # Return the list of possible moves
        return moves

    def DFS(self, current_state, visited, path):
        ##Checks if the current state is the goal state, calls check_goal with its parameters
        if self.check_goal(self.goal_state, current_state, len(visited), self.random_state):
            print("Goal State Reached")
            return path
        #add the current state to the set of visited states
        visited.add(str(current_state))
        ##iterate over all moves
        for move in self.get_possible_moves(current_state):
            # Make a copy of the current state using deepcopy
            new_state = copy.deepcopy(current_state)
            #Get the location of the empty cell in the new state and the location in which to move it to
            i, j = self.get_empty_location(current_state)
            i2, j2 = move
            ##Swap the values of the empty cell and the new location creating a new_state
            new_state[i][j], new_state[i2][j2] = new_state[i2][j2], new_state[i][j]
            ##If this new state is visited, we will end up printing no solution
            if str(new_state) not in visited:
                result = self.DFS(new_state, visited, path + [move])
                #If a solution has been reached return the path
                if result is not None:
                    return result
        ##If no solution is found, return that there is no solution for the random puzzle state
        return "No Solution"


class BreadthFirstSearch:
    ##Take a random state and goal state as parameters along with an empty path
    def __init__(self, random_state, goal_state, current_state, path):
        self.random_state = random_state
        self.goal_state = goal_state
        self.current_state = current_state
        self.path = path
##Function to check whether the goal state is equal to the random state
    def check_goal(self, goal_state, current_state, num_nodes):
        ##Convert our 3x3 nested list into a string 3x3 grid
        current_state_str = "\n".join([" ".join([str(x) for x in row]) for row in current_state])
        ##if all lines in current state are == to the same index in goal state then the goal has been reached and return the
        # final number of nodes along with the path
        if all([current_state[i][j] == goal_state[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(num_nodes) + "\nPath: " + str(
                self.path))
            print("Goal State: \n" + current_state_str + "\n")
            return True
        else:
            ##print the current state of the puzzle solution, this kind of functions as the path if each step from beginning to end was analyzed
            print(
                "Current state:\n" + current_state_str + "\nNodes visited so far:" + str(num_nodes) + "\nPath: " + str(
                    self.path))
            return False
            #function to return the row and column for the position of '-'
    def get_empty_location(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == "-":
                    return i, j
    # function to return the list of valid moves of the empty location from the current state
    def get_possible_moves(self, state):
        moves = []
        i, j = self.get_empty_location(state)
        if i > 0:
            moves.append((i - 1, j))
        if i < 2:
            moves.append((i + 1, j))
        if j > 0:
            moves.append((i, j - 1))
        if j < 2:
            moves.append((i, j + 1))
        return moves
    #method to perform BFS, start parameter = initial state of the puzzle
    def BFS(self, start):
        #create empty set to keep track of states that have been visited
        visited = set()
        #use queue data structure with our initial state and an empty path
        queue = [(start, [])]
        #while the queue is not empty
        while queue:
            #get the next state and path from the queue
            current_state, path = queue.pop(0)
            # if the current state has not been visited
            if str(current_state) not in visited:
                # then mark is as visited by adding it to the set
                visited.add(str(current_state))
                #check if the current state is = to the goal state
                if self.check_goal(self.goal_state, current_state, len(path)):
                    # return the path taken to the goal if so, (here is where the state path instead of get possible moves path is)
                    return path
                #for each possible move from the current state
                for move in self.get_possible_moves(current_state):
                    #make a copy of the current state and the empty location of the current state
                    new_state = copy.deepcopy(current_state)
                    i, j = self.get_empty_location(current_state)
                    i2, j2 = move
                    # add the new state and path to the end of the queue
                    new_state[i][j], new_state[i2][j2] = new_state[i2][j2], new_state[i][j]
                    queue.append((new_state, path + [move]))
                    #if a goal state is never reached return no solution
        return "No Solution"
# The main thing that makes this a BFS is because we use the Queue data structure for our search space


##Takes a different State randomization and goal_state than the above two searching algorithms
class UniformCostSearch:
    ##Initialzize UCS with a random state goal state and current state, which is another random state
    def __init__(self, random_state, goal_state, current_state):
        self.random_state = random_state
        self.goal_state = goal_state
        self.current_state = current_state
        #set the nodes visited to 0, we use this method to store the amount of nodes visited
        self.nodes_visited = 0
#Same as previous check goal functions UPDATE COMMENTS BEFORE POSTING TO GIT
    def check_goal(self, current_state, numNodes):
        current_state_str = "\n".join([" ".join([str(x) for x in row]) for row in current_state])
        if all([current_state[i][j] == self.goal_state[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(numNodes))
            print("Goal State: \n" + current_state_str + "\n")
            return True
        else:
            print("Current state:\n" + current_state_str + "\nNodes visited so far:" + str(numNodes) + "\n")
            return False
#Get the empty location NOW REPRESENTED BY 0
    def get_empty_location(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)
#this combines the previous get possible moves, with the way that we used to find successors inside of BFS and DFS
#assuming in the future we wanted to simplify the code, we could use this as a static method for this state type
    def get_possible_moves(self, state):
        #Initialize an empty list sucessors
        successors = []
        # Get the location of the empty cell that is represented by 0 in the current state
        i, j = self.get_empty_location(state)
        # Loop through all possible moves (up down left and right)
        for move_i, move_j in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            ##check if the move is valid, by making sure is not out of bounds of the puzzle
            ##this prevents an array index out of bound exception
            if 0 <= i + move_i < 3 and 0 <= j + move_j < 3:
                #creates a new state by swapping the 0 cell with a new cell
                new_state = [row[:] for row in state]
                new_state[i][j], new_state[i + move_i][j + move_j] = new_state[i + move_i][j + move_j], \
                                                                     new_state[i][j]
                ##append the new state to the list of sucessors and give it a cost of 1
                successors.append((new_state, 1))
                #return the list of possible moves
        return successors

    def UCS(self):
        # Priority queue implementation of UCS
        #Initialize a priority queue with an initial state and empty path
        hq = [(0, self.current_state, [])]
        #convert the priority queue to a heap queue
        heapq.heapify(hq)
        # store visited states as a set using tuples, they cannot be stored as lists
        # tuples are immutable and lists are unhashable so they will not work in UCS
        # with our current implementation as the costs cannot be compared with a list
        visited = set([tuple(row) for row in self.current_state])
        #Initialize the cost of the initial state to 0
        cost_so_far = {tuple(map(tuple, self.current_state)): 0}
        #while the priority queue is not empty
        while hq:
            #pop the state with the lowest cost
            _, current, path = heapq.heappop(hq)
            ##check whether the popped state is the goal state
            if self.check_goal(current, len(visited)):
                #return the path if the popped state is the goal state
                return path
            #Loop through the possible moves from the current state
            for new_state, cost in self.get_possible_moves(current):
                #find the location of the empty tile (0)
                new_i, new_j = self.get_empty_location(new_state)
                #create a tuple with the move made
                move = (new_i, new_j)
                #calculate the cost of the new state
                new_cost = cost_so_far[tuple(map(tuple, current))] + cost
                #if the new state hasnt been visited or the cost is less than the current cost of the new state
                if tuple(map(tuple, new_state)) not in cost_so_far or new_cost < cost_so_far[
                    tuple(map(tuple, new_state))]:
                    #update the cost
                    cost_so_far[tuple(map(tuple, new_state))] = new_cost
                    #initialize new path to the new state + our move
                    new_path = path + [move]
                    #Add the new state to the priority queue
                    new_visited = set([tuple(row) for row in new_state])
                    heapq.heappush(hq, (new_cost, new_state, new_path))
                    #add visited states to the visited set
                    visited |= new_visited
                    #If the starting state can not be transformed into a goal state then, return no solution
        return "No Solution"
##UCS using a priority queue for its implementation
##however, A* also uses a priority queue, the main difference is that A*
#uses heuristics while UCS does not.
#However we have our uniform cost function, each edge has a fixed cost
#since he have no heuristics we can only see as far as the next node

##A* algorithm implementation
class AStar:
    ##Here we Initialize the same states, along with nodes visited to calculate the cost
    def __init__(self, random_state, goal_state, current_state):
        self.random_state = random_state
        self.goal_state = goal_state
        self.current_state = current_state
        self.nodes_visited = 0
##Same as the other checkGoal functions
    def check_goal(self, current_state, num_nodes):
        current_state_str = "\n".join([" ".join([str(x) for x in row]) for row in current_state])
        if current_state == self.goal_state:
            print("Current State is equal to goal_state. Current State:\n" + current_state_str + "\n")
            print("Goal State:\n" + str(self.goal_state) + "\n")
            return True

        if all([current_state[i][j] == self.goal_state[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(num_nodes))
            return True
        else:
            print("Current state:\n" + current_state_str + "\nNodes visited so far:" + str(num_nodes) + "\n")
            return False
#get location of 0
    def get_empty_location(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)
# get possible moves from current state
    def get_possible_moves(self, state):
        successors = []
        i, j = self.get_empty_location(state)
        for move_i, move_j in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if 0 <= i + move_i < 3 and 0 <= j + move_j < 3:
                #create copy of the current state and swap values
                new_state = [row[:] for row in state]
                new_state[i][j], new_state[i + move_i][j + move_j] = new_state[i + move_i][j + move_j], \
                                                                     new_state[i][j]
                #add the new state to the successors with a cost of 1
                successors.append((new_state, 1))
        return successors
#manhattan distance heuristic, used to help find the cheapest way to reach the goal state without overestimating
    def manhattan_distance(self, state):
        ##Calculates the Manhattan distance between the current state and the goal state
        distance = 0
        ##If empty location is found within the bounds of the state continue
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    continue #skip calculating the distance in the empty cell
                # calculate the goal posiiton of the current number
                row, col = divmod(state[i][j] - 1, 3)
                goal_row, goal_col = divmod(self.goal_state[i][j] - 1, 3)
                #add manhattan distance to the total distance for our heuristic
                distance += abs(row - i) + abs(col - j)
        return distance

    def A_star(self):
        # Priority queue implementation of A*
        #Initialize the priority queue with only the initial state passed as current state\
        # the priority of our initial state is the sum of the manhattan distance and the cost
        #set the path to an empty list
        hq = [(self.manhattan_distance(self.current_state), self.current_state, [],
               set([tuple(row) for row in self.current_state]))]
        heapq.heapify(hq)
        #Initialize variable to keep track of the cost of each state that has been visited
        cost_so_far = {tuple(map(tuple, self.current_state)): 0}
#Loop until we have an empty priority queue
        while hq:
            #Pop the state from the highest priority from the priority queue
            _, current, path, visited = heapq.heappop(hq)
            #If the popped state is the goal state, then return the path (should be state path)
            if self.check_goal(current, len(visited)):
                return path
            #Generate all possible moves from the current state by callign get_possible_moves
            for new_state, cost in self.get_possible_moves(current):
                new_i, new_j = self.get_empty_location(new_state)
                move = (new_i, new_j)
                new_cost = cost_so_far[tuple(map(tuple, current))] + cost
            #If the new state is not present in the tuple or if the cost to get ther eis less
                if tuple(map(tuple, new_state)) not in cost_so_far or new_cost < cost_so_far[
                    tuple(map(tuple, new_state))]:
                    #update the cost to reach the new state and add it to the priority queue ^
                    cost_so_far[tuple(map(tuple, new_state))] = new_cost
                    #Update the patht o the new state
                    new_path = path + [move]
                    # Add the new state to the priority queue and assign it a priority of the sum of the manhattan distance heurisitic and add it accordingly
                    new_visited = set([tuple(row) for row in new_state])
                    heapq.heappush(hq, (
                        new_cost + self.manhattan_distance(new_state), new_state, new_path, visited | new_visited))
        #If the priority queue is empty without finding a goal state then, return no solution
        return "No Solution"

##This is the same as the UCS algorithm earlier with only minor differences, therefore documentation will be light
class UniformCostSearchWithHeuristics:
    def __init__(self, random_state, goal_state, current_state):
        self.random_state = random_state
        self.goal_state = goal_state
        self.current_state = current_state
        self.nodes_visited = 0

    def check_goal(self, current_state, num_nodes):
        current_state_str = "\n".join([" ".join([str(x) for x in row]) for row in current_state])
        if all([current_state[i][j] == self.goal_state[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(num_nodes))
            print("Goal State: \n" + current_state_str + "\n")
            return True
        else:
            print("Current state:\n" + current_state_str + "\nNodes visited so far:" + str(num_nodes) + "\n")
            return False

    def get_empty_location(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)

    def get_possible_moves(self, state):
        successors = []
        i, j = self.get_empty_location(state)
        for move_i, move_j in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if 0 <= i + move_i < 3 and 0 <= j + move_j < 3:
                new_state = [row[:] for row in state]
                new_state[i][j], new_state[i + move_i][j + move_j] = new_state[i + move_i][j + move_j], \
                                                                     new_state[i][j]
                successors.append((new_state, 1))
        return successors
#Purposefully used a weak version of the manhattan distance heuristic to maintain consistency
    # in the difference between UCS and A*
    def h(self, state):
        # Manhattan distance heuristic
        # Instead of using the absolute difference between row and column positions
        # we use a different formula
        dist = 0
        #Although I wanted this one to be less efficent we ended up with just a different way to perform it
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    row = (state[i][j] - 1) // 3
                    col = (state[i][j] - 1) % 3
                    dist += abs(i - row) + abs(j - col)
        return dist

    def ucs_manhattan(self):
        # Priority queue implementation of UCS with Manhattan distance heuristic
        hq = [(0, self.current_state, [])]
        #heapify the priority queue
        heapq.heapify(hq)
        #Initialize the set for visited with the currentState
        visited = set([tuple(row) for row in self.current_state])
        #Initialize the cost so far to 0 in the dictionary
        cost_so_far = {tuple(map(tuple, self.current_state)): 0}
        while hq:
            #pop the state from the lowest cost from the priority queue and check if it equals the goal state
            _, current, path = heapq.heappop(hq)
            if self.check_goal(current, len(visited)):
                #return the path that should be state path
                return path
            #Find all moves from the current state by calling method
            for new_state, cost in self.get_possible_moves(current):
                new_i, new_j = self.get_empty_location(new_state)
                move = (new_i, new_j)
                new_cost = cost_so_far[tuple(map(tuple, current))] + cost
                if tuple(map(tuple, new_state)) not in cost_so_far or new_cost < cost_so_far[
                    tuple(map(tuple, new_state))]:
                    cost_so_far[tuple(map(tuple, new_state))] = new_cost
                    new_path = path + [move]
                    new_visited = set([tuple(row) for row in new_state])
                    #Add the new state to the priority queue with a cost equal to the cost so far + the manhattan distance heuristic
                    heapq.heappush(hq, (new_cost + self.h(new_state), new_state, new_path))
                    visited |= new_visited
        return "No Solution"

##This combines the manhattan distance in the heuristic
# With a misplaced tiles heuristic
# implemented by priority queue
# Will be documenting the different parts
class AStar_With_Hueristics:
    def __init__(self, random_state, goal_state, current_state):
        self.random_state = random_state
        self.goal_state = goal_state
        self.current_state = current_state
        self.nodes_visited = 0

    def check_goal(self, current_state, num_nodes):
        current_state_str = "\n".join([" ".join([str(x) for x in row]) for row in current_state])
        if current_state == self.goal_state:
            print("Current State is equal to goal_state. Current State:\n" + current_state_str + "\n")
            return True
        if all([current_state[i][j] == self.goal_state[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(num_nodes))
            print("Goal State: \n" + current_state_str + "\n")
            return True
        else:
            print("Current state:\n" + current_state_str+ "\nNodes visited so far:" + str(num_nodes) + "\n")
            return False

    def get_empty_location(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)
#Reused old get_possible_moves method, due to it being simple and reliable
#However successor states will have to be made in the A* algorithm
    def get_possible_moves(self, state):
        moves = []
        i, j = self.get_empty_location(state)
        if i > 0:
            moves.append((i - 1, j))
        if i < 2:
            moves.append((i + 1, j))
        if j > 0:
            moves.append((i, j - 1))
        if j < 2:
            moves.append((i, j + 1))
        return moves
#Finds the sum of the vertical and horizantal distances for each tile
    # and the position they are desired to be in
    # takes a state as input
    def manhattan_distance(self, state):
        #Calculates the manhattan distance for the current state
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    row = (state[i][j] - 1) // 3
                    col = (state[i][j] - 1) % 3
                    distance += abs(i - row) + abs(j - col)
        return distance
    #Takes a current state as input
    def misplaced_tiles(self, state):
        #Calculates the misplaced_tiles heuristic for the current state
        #iterates over each tile in the state, if the tile in the current state is different from the goal state
        #the count of misplaced tiles is incremented and the function returns it
        count = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != self.goal_state[i][j]:
                    count += 1
        return count

    def manhattan_misplaced_distance(self, state):
        #we take the integer values, using the current state as input
        # and create a combined heuristic value for manhattan distance and misplaced tiles
        # this is stacking heuristics
        manhattan = self.manhattan_distance(state)
        misplaced = self.misplaced_tiles(state)
        # combine the Manhattan distance and misplaced tiles heuristics
        combined = manhattan + misplaced
        #return the combined heuristic
        return combined

    def A_Star_heuristics(self):
        # Priority queue implementation of A*
        #create a priority queue with heuristic of intiial state, and empty path and set
        hq = [(self.manhattan_misplaced_distance(self.current_state), self.current_state, [],
               set([tuple(row) for row in self.current_state]))]
        heapq.heapify(hq)
        #Create cost thus far with dictionary
        cost_so_far = {tuple(map(tuple, self.current_state)): 0}
#Iterate until the queue is empty
        while hq:
            #get the state witht he lowest combined heuristic value and pop the value
            #then check if it is equal to the goal state
            _, current, path, visited = heapq.heappop(hq)
            if self.check_goal(current, len(visited)):
                return path
            #iterate of all possible moves of the empty tile
            #creating a copy with deepcopy to apply the move
            for move in self.get_possible_moves(current):
                new_state = copy.deepcopy(current)
                #get the empty location for the current tile and swap to create new state
                i, j = self.get_empty_location(current)
                i2, j2 = move
                #create new state
                new_state[i][j], new_state[i2][j2] = new_state[i2][j2], new_state[i][j]
                # calculate the cost to reach the new state
                new_cost = cost_so_far[tuple(map(tuple, current))] + 1
                #check if the new state has been visited or has a lower cost than previous visits in the set
                if tuple(map(tuple, new_state)) not in cost_so_far or new_cost < cost_so_far[
                    tuple(map(tuple, new_state))]:
                    cost_so_far[tuple(map(tuple, new_state))] = new_cost
                    new_path = path + [move]
                    new_visited = set([tuple(row) for row in new_state])
                    #opush the new state into the priority queue with an updated cost and path
                    heapq.heappush(hq, (new_cost + self.manhattan_misplaced_distance(new_state), new_state, new_path,
                                        visited | new_visited))
        #if solution has been found print no solution
        return "No solution"

##Completely new class with no reused methods based on new psuedocode
class IterativeDeepeningSearch:
    #set max depth to (9!/2) IT WOULD TAKE FOREVER IF IT WAS NECESSARY
    def __init__(self, random_state, goal_state, max_depth=181440):
        self.random_state = random_state
        self.goal_state = goal_state
        self.max_depth = max_depth
##takes the current state visited set path and current depth as arguments
    def dfs(self, current, visited, path, depth):
        #if the depth is less or = to 0 return none
        if depth <= 0:
            return None
        #add the current state to the visited set
        visited.add(str(current))

        if current == self.goal_state:
            path.append(current)
            return path
        #check all neighbors of the original state
        for neighbor in self.get_neighbors(current):
            #If not in the visited list append the path to the neighbor
            if str(neighbor) not in visited:
                path.append(neighbor)
                result = self.dfs(neighbor, visited, path, depth - 1)
                if result is not None:
                    return result
                #pop from the stack
                path.pop()

        return None

    def get_neighbors(self, current):
        neighbors = []
        for move in self.get_possible_moves(current):
            neighbor = self.get_next_state(current, move)
            neighbors.append(neighbor)
        return neighbors
#yet another get_possible_moves function, only difference is the variable names
    #i,j could be used however I find this more readable in hindsight
    def get_possible_moves(self, state):
        moves = []
        empty_row, empty_col = self.get_empty_location(state)
        if empty_row > 0:
            moves.append((empty_row - 1, empty_col))
        if empty_row < 2:
            moves.append((empty_row + 1, empty_col))
        if empty_col > 0:
            moves.append((empty_row, empty_col - 1))
        if empty_col < 2:
            moves.append((empty_row, empty_col + 1))
        return moves
#get empty location for '-' instead of 0 like a little previously
    def get_empty_location(self, state):
        for row in range(3):
            for col in range(3):
                if state[row][col] == '-':
                    return row, col

    def get_next_state(self, current, move):
        row, col = self.get_empty_location(current)
        move_row, move_col = move
        new_state = copy.deepcopy(current)
        new_state[row][col] = new_state[move_row][move_col]
        new_state[move_row][move_col] = '-'
        return new_state
##Similar to the checkGoal method however implemented along with IDS function to incrememnt the depth by once each recursive call
    def run(self):
        #Initialize set and path
        #and result boolean flag
        visited = set()
        path = []
        result = None
        for depth in range(1, self.max_depth + 1):
            #Clear the visited set and path list, since we are searching a level deeper
            visited.clear()
            path.clear()
            #calls dfs function recursively
            result = self.dfs(self.random_state, visited, path, depth)
            if result is not None:
                print("Solution found at depth", len(result) - 1)
                print("Path:", result)
                #return the solution path
                return result
            else:
                #print that no solution has been found at the current depth
                print("No solution found at depth " + str(depth))
                #add the current state
                self.random_state_str = "\n".join([" ".join([str(x) for x in row]) for row in self.random_state])
                print("Current state: \n"+ self.random_state_str)
        print("Reached maximum depth of", self.max_depth)
        return "No Solution"
#uses a stack for IDS

class Controller:
    #Call the Random Selector class to randomly select a searching algorithm on run
    #This also allows for more accurate runtime calculation, could also be done through seperating the programs
    #however I feel like I am implementing an agent
    #This is a conspiracy howeverm I dont believe the chance of getting each algorithm is the same, although theoretically it should be
    RS = RandomSelector().random_selector()
    #If RS == 1 then call DFS Recursive method (useless)
    if RS == 1:
        sr = State().state_randomizer()
        gs = State().goal_state()
        dt = DepthFirstSearch(sr, gs, sr)
        visited = set()
        path = []
        resultDFS = dt.DFS(sr, visited, path)
        print(resultDFS)
        print("completed 8-Puzzle in DFS Recursive")
    elif RS == 2:
        sr = State().state_randomizer()
        gs = State().goal_state()
        bfs = BreadthFirstSearch(sr, gs, sr, [])
        resultBFS = bfs.BFS(sr)
        print(resultBFS)
        print("Completed 8-Puzzle in BFS")
    elif RS == 3:
        randomState = State().state_randomizer_int()
        goal_state = State().goal_state_int()
        ucs = UniformCostSearch(randomState, goal_state, randomState)
        path = ucs.UCS()
        print(path)
        print("Completed 8-Puzzle in UCS")
    elif RS == 4:
        randomState = State().state_randomizer_int()
        goal_state = State().goal_state_int()
        starA = AStar(randomState, goal_state, randomState)
        path = starA.A_star()
        print(path)
        print("Completed 8-Puzzle in A*")
    elif RS == 5:
        randomState = State().state_randomizer_int()
        goal_state = State().goal_state_int()
        ucs = UniformCostSearchWithHeuristics(randomState, goal_state, randomState)
        path = ucs.ucs_manhattan()
        print(path)
        print("Completed 8-Puzzle in UCS with heuristics")
    elif RS == 6:
        randomState = State().state_randomizer_int()
        goal_state = State().goal_state_int()
        starA = AStar_With_Hueristics(randomState, goal_state, randomState)
        path = starA.A_Star_heuristics()
        print(path)
        print("Completed 8-Puzzle in A* with heuristics")
    elif RS == 7:
        sr = State().state_randomizer()
        gs = State().goal_state()
        ids = IterativeDeepeningSearch(sr, gs)
        resultIDS = ids.run()
        print(resultIDS)
        print("completed 8-Puzzle in DFS Iterative")
    end_time = time.time()
    runtime = (end_time - RandomSelector.start_time) * 1000
    print("Program ran in " + str(runtime))

##Changelog:
# Constructed DFS recursive class, despite me seeing the implementation in the textbook be IDS, scrapped the implementation
# Using the working methods from DFS created BFS
# Edited the State class to contain a random state with only integers, and a goal state containing only integers, this prevents the Type Error and subsequent nonsubscriptable errors, when trying to compare '-' to an int
# previous problem may have been fixable by using an actual blank space, however I was too far along
# Made new get_possible_moves function for UCS and A* based on some inspiration I saw from git after various errors and debugging
# changed to tuple usage with priority queues over just lists to prevent unhashable errors
# This effectively combined the way that we previous got successors inside our searching algorithm's and got it inside one method, calling it when necessary
# returned to original get_possible_moves function A* with Heuristics
# Changed Check Goal to not compare the initial state random state, as it was redundant
# made new class for IDS without reusing previous methods or logic, this was to attempt to prevent some of the errors plaguing my previous code
# changed the format of all variables and classes to PEP 8 Styles, as well as giving every variable an easily understandable name
# I developed a habit of trying to make my variable names unique, however with me preparing for more professional projects I should make sure they are easily understandable
# Changed check_goal function to return a proper 3x3 grid for each state, however with minimal time left I did not do the same for the path
# Actually added the heuristics to UCS_with heuristics, I copy and pasted from UCS, without implementing the heuristic