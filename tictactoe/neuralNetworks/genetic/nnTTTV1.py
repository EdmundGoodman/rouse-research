#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5

#Add bias node, to allow better first move?

#Add probability gradient
#Return to playing itself

import os, random, itertools, copy, time
from collections import Counter
import numpy as np
from visualisation import *

os.system('clear')
np.seterr(all='ignore')
np.random.seed(2000) #2000
random.seed(2000)

class NeuralNetwork(object):
    def __init__(self,x,y,z, weightDP=5, weights=None):
        #Define hyperparameters
        self.inputSize = x
        self.hiddenSize = y
        self.outputSize = z
        self.weightDP = weightDP

        if weights is None:
            self.generateRandomWeights()
        else:
            self.setWeights(weights)


    def generateRandomWeights(self):
        def randTuple(x,y, r=5, m=0.01):
            return np.round(np.random.randn(x,y), r) * m

        #Generate random weights
        self.W1 = randTuple(self.inputSize, self.hiddenSize[0])
        self.W2 = [
            randTuple(
                self.hiddenSize[i],
                self.hiddenSize[i+1]
            ) for i in range(0, len(self.hiddenSize)-1)]
        self.W2.append(randTuple(self.hiddenSize[-1], self.outputSize))

    def setWeights(self, weights):
        #Set the weights, as specified
        self.W1 = weights[0]
        self.W2 = weights[1]

    def forwardpropagate(self, X):
        #Forward propagate data though the network
        self.z, self.a = [], []
        #From input to hidden layer
        self.z.append( np.dot(X, self.W1) )
        self.a.append( self.activation(self.z[-1]) )
        #Through the hidden layers
        for i in range(len(self.W2) - 1):
            self.z.append( np.dot(self.a[-1], self.W2[i]) )
            self.a.append( self.activation(self.z[-1]) )
        #From hidden to output layer
        self.z.append( np.dot(self.a[-1], self.W2[-1]) )
        self.a.append( self.activation(self.z[-1]) )
        #Return the final value of the output layer
        return self.a[-1]

    @staticmethod
    def activation(z, deriv=False):
        #Apply the activation function to scalar, vector, or matrix
        if not deriv:
            return np.tanh(z)#1/(1+np.exp(-z))
        else:
            return z*(1-z)

    def __repr__(self):
        output = "A {}-{}-{} fully connected network with weights:\n{}\n{}".format(
            self.inputSize,
            self.hiddenSize,
            self.outputSize,
            self.W1,
            self.W2,
        )
        return output




    def generateLayers(self):
        self.layers = []
        self.layers.append( Layer(self, self.outputSize, self.W2[-1]) )
        for i in reversed(range(1,len(self.hiddenSize))):
            self.layers.append( Layer(self, self.hiddenSize[i], self.W2[i-1]) )
        self.layers.append( Layer(self, self.hiddenSize[0], self.W1) )
        self.layers.append( Layer(self, self.inputSize, None) )

    def drawNetworkLayout(self, weightMultiplier=1):
        self.generateLayers()
        for layer in self.layers:
            layer.draw(weightMultiplier)
        plt.axis('scaled')
        plt.show()

    def drawHintonDiagram(self):
        diagram = HintonDiagram()
        diagram.show([self.W1] + self.W2)




class Tictactoe:
    def __init__(self):
        #Set the game variables
        self.players = ["O", "X"]
        self.resetGameVariables()

    def resetGameVariables(self):
        #Reset the board, the current player, and the winner
        self.board = [["_" for _ in range(3)] for _ in range(3)]
        self.playerNum = 0
        self.winner = False

    def checkIfWon(self):
        #Return the winner (Name/Draw/False)
        b = self.board
        for p in self.players:
            #All the win conditions for tic tac toe
            if (b[2][0] == p and b[2][1] == p and b[2][2] == p): return p
            elif (b[1][0] == p and b[1][1] == p and b[1][2] == p): return p
            elif (b[0][0] == p and b[0][1] == p and b[0][2] == p): return p
            elif (b[2][0] == p and b[1][0] == p and b[0][0] == p): return p
            elif (b[2][1] == p and b[1][1] == p and b[0][1] == p): return p
            elif (b[2][2] == p and b[1][2] == p and b[0][2] == p): return p
            elif (b[2][0] == p and b[1][1] == p and b[0][2] == p): return p
            elif (b[2][2] == p and b[1][1] == p and b[0][0] == p): return p
        if set([x for sublist in self.board for x in sublist]).isdisjoint(set("_")):
            return "Draw"
        return False

    def displayBoard(self):
        #Print the board
        for y in self.board:
            print(y)
        print()

    def makeMove(self, player, move):
        #Make the move
        self.board[move[1]][move[0]] = player

    def isMoveValid(self, position):
        #Check if the move is valid
        if 0<=position[0]<=2 and 0<=position[0]<=2:
            if self.board[position[1]][position[0]] == "_":
                return True
        return False

    def aiNeuralMove(self, nn):
        #Use the neural network to generate move
        iL = np.array(self.board, dtype=str).flatten()
        iL = [{"X": 2, "O": 1, "_": 0}.get(item,item) for item in iL]
        X = np.nan_to_num(iL/np.amax(iL, axis=0))
        yHat = nn.forwardpropagate(X)
        while True:
            indexMax = np.argmax(yHat)
            move = [indexMax%3, indexMax//3]
            if self.isMoveValid(move):
                return move
            else:
                yHat[indexMax] = 0

    def getEmptySquarePositions(self):
        positions = []
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] == "_":
                    positions.append((x,y))
        return positions

    def getBestMove(self, depth=0):
        currentBoard = copy.deepcopy(self.board)
        selfPlayer = self.players[self.playerNum]
        otherPlayer = self.players[(self.playerNum+1)%2]

        if currentBoard ==  [["_" for _ in range(3)] for _ in range(3)]:
            return (0,0) #random.choice([(0,0), (2,0), (0,2), (2,2)])

        winner = self.checkIfWon()
        if winner is not False:
            if winner == selfPlayer:
                return 2
            elif winner == otherPlayer:
                return -2
            else: #Draw
                return 0

        moveWeights = {}
        for position in self.getEmptySquarePositions():
            if depth%2==0:
                self.makeMove(selfPlayer, position)
            else:
                self.makeMove(otherPlayer, position)
            moveWeights[position] = self.getBestMove(depth+1)

            self.board = copy.deepcopy(currentBoard)

        if depth%2==0: #If it is the ai playing, play best move for the ai
            bestMove = max(moveWeights, key=lambda key: moveWeights[key])
        else: #Otherwise, play the best move the other player can make (worst move for the ai)
            bestMove = min(moveWeights, key=lambda key: moveWeights[key])
        bestWeight = moveWeights[bestMove]

        if depth == 0:
            return bestMove
        else:
            return bestWeight

    def humanMove(self):
        while True:
            move = [int(x) for x in input("Position x,y: ").split(',')]
            if self.isMoveValid(move):
                return move
            else:
                print("Invalid Move")

    def getRandomMove(self):
        return random.choice(self.getEmptySquarePositions())

    def play(self, nn):
        while True:
            self.displayBoard()
            if not self.playerNum:
                print("AI move: ")
                move = self.aiNeuralMove(nn)
            else:
                print("Player move: ")
                #move = self.getBestMove()
                move = self.humanMove()
            self.makeMove(self.players[self.playerNum], move)
            self.playerNum = (self.playerNum+1)%2
            self.winner = self.checkIfWon()
            if self.winner:
                break
        self.displayBoard()
        return self.winner

    def competeNetworks(self, nns):
        #Train the neural network by playing to against each other
        while True:
            move = self.aiNeuralMove(nns[self.playerNum])
            self.makeMove(self.players[self.playerNum], move)
            self.playerNum = (self.playerNum+1)%2
            self.winner = self.checkIfWon()
            if self.winner:
                break
        return self.winner

    def testNetworks(self, nn, noTests=20):
        noWins = 0
        for i in range(20):
            while True:
                if not self.playerNum:
                    move = self.aiNeuralMove(nn)
                else:
                    move = self.getRandomMove()
                self.makeMove(self.players[self.playerNum], move)
                self.playerNum = (self.playerNum+1)%2
                self.winner = self.checkIfWon()
                if self.winner:
                    break
            if self.winner != self.players[1]:
                noWins += 1
            game.resetGameVariables()

        return noWins/noTests







def getNewNetworks(numNetworks):
     for _ in range(numNetworks):
         yield NeuralNetwork(*sizes)

def getWinner(pairIn):
    global noDraws, noWins, noLosses
    winners = []
    for i,pair in enumerate([pairIn, pairIn[::-1]]):
        winner = game.competeNetworks(pair)
        if winner != "O":
            winners.append(pair[i])
            noWins += 1
        elif winner == "Draw":
            if genNum > allowDrawNum:# and random.random() < 0.5:
                winners.append(pair[i])
            noDraws += 1
        else:
            noLosses += 1
        game.resetGameVariables()
    return winners

def breedNetworks(nns, percent=100):
    #Breed the networks together, by randomly taking data from each of their parents
    #Somewhat analogous to "crossing over" in biological genetic systems
    def select(a, b, c, d):
        for i in range(0, len(a)):
            if type(a[i]) is list:
                z = select(a[i], b[i], [], d+1)
                c.append(z)
            else:
                c.append( random.choice([a[i]+b[i]]) )
        return c

    nnWeights = [[nn.W1, nn.W2] for nn in nns]
    breedingPairs = list(itertools.combinations(nnWeights, 2))
    random.shuffle(breedingPairs)
    breedingPairs = breedingPairs[:int( (percent/100)*len(breedingPairs) )]

    bredWeights = [select(b[0], b[1], [], 1) for b in breedingPairs]
    nns = [NeuralNetwork(*sizes, weights=bredWeights[i]) for i in range(len(bredWeights))]
    return nns


def playWinner(nn):
    game = Tictactoe()
    result = game.play(nn)
    if result == "Draw":
        print("Draw!")
    else:
        print("{} won!".format(result))


print("Starting training")

#Initialise the game
game = Tictactoe()

#Meta parameters
sizes = [9, [3,3], 9] #[9,[3,3],9]

numGenerations, initialSize, concurrentLim = 10, 100, 20 #20
finalSizes = [75] * (numGenerations-1) + [1]
breedPercents = [50] * numGenerations
newNetworks = [1500] * numGenerations

startTime = time.time()


cohort, cohortSize = getNewNetworks(initialSize), initialSize
stopFlag = False

try: #Train until KeyboardInterrupt or entire program is complete
    for genNum in range(numGenerations):


        if genNum != 0:
            print("Making new cohort")
            newCohort = cohort[:]
            newCohort.extend(breedNetworks(cohort, percent=breedPercents[genNum]))
            random.shuffle(newCohort)
            cohort, cohortSize = newCohort[:], len(newCohort)


        print("Whittling down cohort: ")
        prevCohortSizes = [None]*concurrentLim
        while cohortSize > finalSizes[genNum]:
            print("Yes")
            winners = []
            pair = [None, None]
            noDraws, noWins, noLosses = 0, 0, 0
            for i,c in enumerate(cohort):
                if i%2==0:
                    pair[0] = c
                else:
                    pair[1] = c

                    winner = getWinner(pair)
                    winners.extend(winner)

            winPercent = round(len(winners)*(100/cohortSize),2)

            cohortSize = len(winners)
            prevCohortSizes = prevCohortSizes[1:] + [cohortSize]

            if cohortSize < 2:
                print("If training failed in this case, try increasing the size of the initial pool")
                stopFlag = True
                break

            cohort = winners[:]
            random.shuffle(cohort)

            #We would hope the number of draws increases as the nets are trained
            #However, on closer inspection, draws are unlikely since both players
            #   optimise for playing first


            print("\tGeneration: {}; {}; W:{};L{};D:{}; {}% {}%".format(
                genNum+1, cohortSize, noWins, noLosses, noDraws, winPercent,
                game.testNetworks(list(cohort)[0], noTests=50)*100
            ))

            if len(set(prevCohortSizes)) == 1:
                print("Concurrent")
                cohort = cohort[:finalSizes[genNum]]
                cohortSize = len(cohort)
                stopFlag = True


        if stopFlag:
            break
except KeyboardInterrupt:
    cohort = winners[:]
finally:
    print("Final network resulting from training: ")
    print(list(cohort)[0])
    os.system("clear")

print("Time taken: {}".format(round(time.time()-startTime, 2)))
bestNetwork = list(cohort)[0]

bestNetwork.drawHintonDiagram() #drawNetworkLayout()
print(game.testNetworks(bestNetwork, noTests=50))
#while True:
#    os.system("clear")
#    playWinner(bestNetwork)
#print("If 'O won!', or 'Draw!', the network is promisingly trained")
