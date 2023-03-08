import random
import copy
class DepthFirstSearch:
    def __init__(self, randomState, goalState, currentState):
        self.randomState = randomState
        self.goalState = goalState
        self.currentState = currentState

    def checkGoal(self, goalState, currentState, numNodes, randomState):
        if currentState == goalState:
            print("Initial State is equal to goalState. Initial State:\n" + str(currentState) + "\n")
            return True
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
