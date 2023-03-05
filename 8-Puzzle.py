import random
import copy


##I will run each algorithm 10 times, which algorithm is selected to sort the state will be selected randomly
##This allows for me to not have to add input while selecting the
class RandomSelector:
    ##We use this to randomly select a searching algorithm on run
    def randomSelector(self):
        randomNumber = random.randint(1, 6)
        if randomNumber == 1:
            print("Implementing DFS")
        elif randomNumber == 2:
            print("Implementing BFS")
        elif randomNumber == 3:
            print("Implementing UCS")
        elif randomNumber == 4:
            print("Implementing A*")
        elif randomNumber == 5:
            print("Implementing Heuristics used in UCS")
        elif randomNumber == 6:
            print("Implementing Heuristics used in A*")
        return randomNumber


class State:
    def stateRandomizer(self):
        grid = [[], [], []]
        goalGrid = State.goalState(self)
        elementList = [i for i in range(1, 9)] + ['-']
        random.shuffle(elementList)
        randomStateGrid = [elementList[i:i + 3] for i in range(0, 9, 3)]
        myStateGrid = randomStateGrid
        return myStateGrid

    def goalState(self):
        goalGrid = [[1, 2, 3], [8, '-', 4], [7, 6, 5]]
        return goalGrid


grid = State().stateRandomizer()

class depthFirstSearch:
    def __init__(self, randomState, goalState, currentState):
        self.randomState = randomState
        self.goalState = goalState
        self.currentState = currentState

    def checkGoal(self, goalState, currentState, numNodes, randomState):
        if currentState == randomState:
            print("Initial State:\n" + str(currentState) + "\n")
            return False
        if all([currentState[i][j] == goalState[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(numNodes))
            print("Goal State: \n"+str(currentState)+"\n")
            return True
        else:
            print("Current state:\n" + str(currentState) + "\nNodes visited so far:" + str(numNodes) + "\n")
            return False

    def getEmptyLocation(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == "-":
                    return (i, j)

    def getPossibleMoves(self, state):
        moves = []
        i, j = self.getEmptyLocation(state)
        if i > 0:
            moves.append((i - 1, j))
        if i < 2:
            moves.append((i + 1, j))
        if j > 0:
            moves.append((i, j - 1))
        if j < 2:
            moves.append((i, j + 1))
        return moves

    def DFS(self, cState, visited, path):
        numNodes = len(visited)
        if self.checkGoal(self.goalState, cState, numNodes, self.randomState):
            print("Goal State Reached")
            return path
        visited.add(str(cState))
        for move in self.getPossibleMoves(cState):
            newState = copy.deepcopy(cState)
            i, j = self.getEmptyLocation(cState)
            i2, j2 = move
            newState[i][j], newState[i2][j2] = newState[i2][j2], newState[i][j]
            if str(newState) not in visited:
                result = self.DFS(newState, visited, path + [move])
                if result is not None:
                    return result
        return None

sr = State().stateRandomizer()
gs = State().goalState()
kt = depthFirstSearch(sr, gs, sr)
visited = set()
path = []
result = kt.DFS(sr, visited, path)
print(result)


class BreadthFirstSearch:
    def __init__(self, randomState, goalState, currentState):
        self.randomState = randomState
        self.goalState = goalState
        self.currentState = currentState

    def checkGoal(self, goalState, currentState, numNodes, randomState):
        if currentState == randomState:
            print("Initial State:\n" + str(currentState) + "\n")
            return False
        if all([currentState[i][j] == goalState[i][j] for i in range(3) for j in range(3)]):
            print("The goal has been reached\nNodes required for completion: " + str(numNodes))
            print("Goal State: \n" + str(currentState) + "\n")
            return True
        else:
            print("Current state:\n" + str(currentState) + "\nNodes visited so far:" + str(numNodes) + "\n")
            return False

    def getEmptyLocation(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == "-":
                    return (i, j)

    def getPossibleMoves(self, state):
        moves = []
        i, j = self.getEmptyLocation(state)
        if i > 0:
            moves.append((i - 1, j))
        if i < 2:
            moves.append((i + 1, j))
        if j > 0:
            moves.append((i, j - 1))
        if j < 2:
            moves.append((i, j + 1))
        return moves

    def BFS(self, start):
        visited = set()
        queue = [(start, [])]
        while queue:
            cState, path = queue.pop(0)
            if str(cState) not in visited:
                visited.add(str(cState))
                if self.checkGoal(self.goalState, cState, len(path)):
                    return path
                for move in self.getPossibleMoves(cState):
                    newState = copy.deepcopy(cState)
                    i, j = self.getEmptyLocation(cState)
                    i2, j2 = move
                    newState[i][j], newState[i2][j2] = newState[i2][j2], newState[i][j]
                    queue.append((newState, path + [move]))
        return None


class UniformCostSearch:
    def UCS(self):
        self.UCS()
