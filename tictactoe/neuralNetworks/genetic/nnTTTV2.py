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
            return 1/(1+np.exp(-z))
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
        #iL = [{"X": 1, "O": -1, "_": 0}.get(item,item) for item in iL]
        #X = np.nan_to_num(iL/np.amax(iL, axis=0))
        X = np.array([0 if x=="X" else 1 for x in iL]+[0 if x=="O" else 1 for x in iL])

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
                move = self.getBestMove()
                #move = self.humanMove()
            self.makeMove(self.players[self.playerNum], move)
            self.playerNum = (self.playerNum+1)%2
            self.winner = self.checkIfWon()
            if self.winner:
                break
        self.displayBoard()
        return self.winner

    def competeNetworks(self, nns):
        #Train the neural network by playing to against each other
        count = 0
        while True:
            if not self.playerNum:
                move = self.aiNeuralMove(nns[self.playerNum])
            else:
                #move = self.aiNeuralMove(nns[self.playerNum])
                move = self.getRandomMove()
            count += 1
            self.makeMove(self.players[self.playerNum], move)
            self.playerNum = (self.playerNum+1)%2
            self.winner = self.checkIfWon()
            if self.winner:
                break
        return self.winner, count









def getNewNetworks(numNetworks):
     for _ in range(numNetworks):
         yield NeuralNetwork(*sizes)

def getWinner(pairIn):
    global noDraws, noWins, noLosses
    winners = []
    for i,pair in enumerate([pairIn, pairIn[::-1]]):
        winner, noMoves = game.competeNetworks(pair)
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
sizes = [18, [6], 9] #[9,[3,3],9]

numGenerations, initialSize, concurrentLim = 20, 50000, 20 #20
finalSizes = [1000] * (numGenerations-1) + [1]
breedPercents = [5] * numGenerations
newNetworks = [0] * numGenerations

startTime = time.time()


cohort, cohortSize = getNewNetworks(initialSize), initialSize
bestNetwork = None
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
            print("\tGeneration: {}; {}; W:{};L{};D:{}; {}%".format(
                genNum+1, cohortSize, noWins, noLosses, noDraws, winPercent
            ))

            if len(set(prevCohortSizes)) == 1:
                print("Concurrent")
                cohort = cohort[:finalSizes[genNum]]
                cohortSize = len(cohort)
                stopFlag = True


        if stopFlag:
            break
except KeyboardInterrupt:
    pass
finally:
    print("Final network resulting from training: ")
    print(list(cohort)[0])
    #os.system("clear")

print("Time taken: {}".format(round(time.time()-startTime, 2)))
list(cohort)[0].drawHintonDiagram() #drawNetworkLayout()
while True:
    #os.system("clear")
    playWinner(list(cohort)[0])
    break
#print("If 'O won!', or 'Draw!', the network is promisingly trained")


"""
A 9-[3, 3]-9 fully connected network with weights:
[[ 0.1574663  0.1136234 -0.0198585]
 [-0.0849317 -0.0476796 -0.0966577]
 [-0.0101242 -0.3601197 -0.255096 ]
 [-0.0062385 -0.1306642  0.1655083]
 [-0.0555675  0.2292235  0.1907182]
 [ 0.1703238  0.0441232  0.1233744]
 [-0.0598033 -0.0876402 -0.1694057]
 [ 0.0338237  0.0050589 -0.0293966]
 [ 0.1367842  0.1496663 -0.004712 ]]
[array([[ 0.2645129, -0.0884783,  0.1420629],
       [ 0.0295464, -0.0206597, -0.0852307],
       [ 0.1332424,  0.0684572,  0.2414184]]), array([[-0.0906984,  0.0935035, -0.
0047206,  0.2617761,  0.0971076,
        -0.240973 ,  0.1031248, -0.006863 ,  0.0659374],
       [ 0.1443628, -0.0659972,  0.1811472,  0.2297462,  0.2039145,
        -0.0659881, -0.2359148, -0.3723173,  0.0523989],
       [-0.0525158, -0.2802466, -0.1005464, -0.0121052,  0.2484192,
         0.1002996,  0.1361125, -0.0449769,  0.1430731]])]
Time taken: 219.5









[[-1.1638578e+00  1.8051106e+00  1.6859493e+00 -1.0399927e+00
   1.2980409e+00  2.1157744e+00]
 [ 1.2478739e+00  2.0284812e+00 -9.3534280e-01 -3.2232465e+00
  -2.2586000e-01  5.2663300e-02]
 [ 2.6961878e+00 -7.7260930e-01 -2.4292337e+00 -1.6130840e-01
  -6.8527740e-01  5.7197820e-01]
 [ 1.1912435e+00  1.4595973e+00 -1.1724044e+00 -2.3425812e+00
   1.3148845e+00 -1.7280215e+00]
 [-1.2496171e+00  2.2303610e-01 -7.1664880e-01  1.0824256e+00
  -1.9445300e-01 -4.6861300e-02]
 [-1.9728369e+00  6.0484130e-01 -3.2781000e-01  1.4365152e+00
   2.1529420e+00  1.2806360e+00]
 [ 4.0891220e-01 -2.2000812e+00  8.2390040e-01  9.0945730e-01
  -9.3097650e-01  2.8786270e-01]
 [-8.1924750e-01 -4.5892420e-01  2.8353560e-01  2.6081534e+00
  -7.9901020e-01 -1.0182006e+00]
 [ 1.1176036e+00 -1.2051811e+00  1.3624933e+00 -1.6757229e+00
   9.6890770e-01 -1.0025549e+00]
 [-1.1001218e+00  1.2452417e+00 -6.4050790e-01  1.3582440e-01
  -1.0828629e+00 -1.8467348e+00]
 [-1.4548318e+00 -3.1016878e+00  4.3021310e-01 -9.6510400e-02
  -2.6099359e+00 -2.7099669e+00]
 [ 1.6334223e+00  1.2416997e+00  3.8728618e+00  1.5396747e+00
  -1.7621680e-01 -2.0162628e+00]
 [-1.3915136e+00  2.6949830e-01  1.7314131e+00  2.3230914e+00
   8.8861450e-01 -5.8829580e-01]
 [-4.7188323e+00 -4.2138880e-01 -2.1070646e+00  1.6003128e+00
  -8.7312250e-01  1.1918396e+00]
 [ 7.3533820e-01  1.1843493e+00  1.2350850e+00 -1.0580006e+00
  -4.9233000e-02  2.2360354e+00]
 [ 2.1352630e-01  4.2673236e+00  3.7398989e+00 -9.4955570e-01
   1.9112073e+00  1.8734686e+00]
 [ 1.4291241e+00  4.2774850e-01  1.2003381e+00  1.0076530e-01
  -1.1251231e+00 -2.2502984e+00]
 [-7.7375850e-01  2.0768955e+00  2.9106000e-03  1.7684866e+00
  -2.5926376e+00 -1.1511336e+00]]
[array([[-1.7513312,  0.432634 , -1.2117421, -2.6898891, -0.912159 ,
         2.8834092,  3.5278883,  0.2131387,  1.3818836],
       [ 0.651933 , -0.3213088, -0.9235251, -0.2116765,  0.8052466,
         1.5259717,  0.9292667,  0.0817785,  3.1535589],
       [ 1.7685053,  1.233487 , -1.5412898, -0.7813653,  0.4924999,
         2.1899694,  2.3665891, -1.7629436,  0.4201571],
       [-1.1819443,  0.8401204,  1.0977968, -0.575506 , -2.0092095,
         1.0509967,  2.7817953, -2.5439172,  1.4935987],
       [ 1.528682 , -0.4605953,  2.1462574, -2.6246099,  1.277102 ,
         1.5390907,  3.1548901, -1.2935067, -0.4417241],
       [ 0.7392313, -0.7152857, -2.4349397,  0.9583006, -1.5070769,
         0.4222706,  0.5718483, -0.0610197, -1.4958503]])]
Time taken: 818.81



A 18-[6]-9 fully connected network with weights:
[[ 0.4153335 -3.0025104  1.6616639 -3.6931547  3.9000929  0.7970014]
 [-4.0041114 -3.6743305  4.2377261 -1.1387222  1.7343479  3.029363 ]
 [ 2.1354516  1.8364949 -2.556392  -2.3947528  1.7524994 -1.0776789]
 [ 1.3538546 -1.0432978 -0.9262871 -3.0763515 -0.6897981 -1.353013 ]
 [-0.181541  -1.2072695 -0.669128   1.9166046  2.8445901  1.6205585]
 [-1.7681091 -1.2377924 -1.4203961 -2.7468826  2.3655738 -2.4389942]
 [-3.77221    1.3841197 -1.0857268 -2.0275345 -1.2621348 -1.9658286]
 [-5.2598743  4.6590385 -6.5268623  0.4675056  0.2247291  0.1077136]
 [ 3.2019261 -1.963637  -2.4900524  0.287934   0.0338498 -0.5233627]
 [ 1.451528  -1.0280325  4.2093361  4.1430863 -0.6754353 -1.282139 ]
 [-0.0803711  0.9824447 -0.9568186  0.4406228 -1.4506944 -2.5391351]
 [ 5.7617517  0.6322337 -1.8161741 -1.1881648 -1.2178658 -3.5774605]
 [ 3.8862871  0.7074262 -2.5606061 -1.6107431  0.8183606  0.1171857]
 [-3.5755361 -5.825033  -0.4463686  3.6954073 -1.23549   -0.2792439]
 [ 1.0501109  1.6881947  0.811597  -2.0545496  3.0480052  2.3538523]
 [-0.2595175 -0.7033047  5.6704417  1.912195  -3.1926438 -1.7740994]
 [ 1.5301952  1.8189009  0.0664511 -0.8079952  0.6913077 -3.0243662]
 [ 1.4040526 -4.9489065 -1.1987397 -2.1416326 -5.9932876  1.3376034]]
[array([[ 1.5063678, -3.59084  ,  3.3867705, -1.9117891, -0.5314472,
         0.2319901,  3.1115531,  5.4275843,  5.9285132],
       [-1.0288427,  1.6057758, -2.7070826, -2.9528002, -1.5507254,
         0.250327 , -4.2282106,  1.7520776,  2.1850557],
       [ 0.0415813, -2.7011491,  1.6573533,  3.8731699, -0.592801 ,
        -0.0473847,  3.7327292, -2.4093102, -0.296301 ],
       [ 0.8657795,  1.1693417,  4.3262422, -3.0880076,  1.6806473,
         0.0693738,  1.4706943,  0.192214 , -0.0112448],
       [ 1.9129207, -1.7674671,  4.294738 ,  0.8434794,  0.3505733,
        -3.8301949,  4.5717714, -3.3898552, -1.033886 ],
       [ 0.209623 , -2.3351512, -1.2225695, -0.7353692,  3.070694 ,
        -2.7572583,  3.2619181,  0.0633609,  0.3294698]])]
Time taken: 413.6
"""





"""
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
            return 1/(1+np.exp(-z))
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
        #iL = [{"X": 1, "O": -1, "_": 0}.get(item,item) for item in iL]
        #X = np.nan_to_num(iL/np.amax(iL, axis=0))
        X = np.array([0 if x=="X" else 1 for x in iL]+[0 if x=="O" else 1 for x in iL])

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
        count = 0
        while True:
            if not self.playerNum:
                move = self.aiNeuralMove(nns[self.playerNum])
            else:
                move = self.aiNeuralMove(nns[self.playerNum])
                #move = self.getRandomMove()
            count += 1
            self.makeMove(self.players[self.playerNum], move)
            self.playerNum = (self.playerNum+1)%2
            self.winner = self.checkIfWon()
            if self.winner:
                break
        return self.winner, count









def getNewNetworks(numNetworks):
     for _ in range(numNetworks):
         yield NeuralNetwork(*sizes)

def getWinner(pairIn):
    global noDraws, noWins, noLosses
    winners = []
    for i,pair in enumerate([pairIn, pairIn[::-1]]):
        winner, noMoves = game.competeNetworks(pair)
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
sizes = [18, [6], 9] #[9,[3,3],9]

numGenerations, initialSize, concurrentLim = 15, 40000, 10 #20
finalSizes = [800] * (numGenerations-1) + [1]
breedPercents = [5] * numGenerations
newNetworks = [0] * numGenerations

startTime = time.time()


cohort, cohortSize = getNewNetworks(initialSize), initialSize
bestNetwork = None
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
            print("\tGeneration: {}; {}; W:{};L{};D:{}; {}%".format(
                genNum+1, cohortSize, noWins, noLosses, noDraws, winPercent
            ))

            if len(set(prevCohortSizes)) == 1:
                print("Concurrent")
                cohort = cohort[:finalSizes[genNum]]
                cohortSize = len(cohort)
                stopFlag = True


        if stopFlag:
            break
except KeyboardInterrupt:
    pass
finally:
    print("Final network resulting from training: ")
    print(list(cohort)[0])
    #os.system("clear")

print("Time taken: {}".format(round(time.time()-startTime, 2)))
list(cohort)[0].drawHintonDiagram() #drawNetworkLayout()
while True:
    #os.system("clear")
    playWinner(list(cohort)[0])
#print("If 'O won!', or 'Draw!', the network is promisingly trained")

"""
