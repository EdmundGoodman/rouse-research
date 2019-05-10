#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5
import numpy as np
import os, random, pickle, math, copy
from visualisation import *

random.seed(1)
np.random.seed(1)

#Add pruning
#Add momentum


class NeuralNetwork(object):
    def __init__(self,x,y,z, Lambda=0, dropout=[0]*3):
        #Define hyperparameters
        self.inputSize = x
        self.hiddenSize = y
        self.outputSize = z
        self.Lambda = Lambda
        self.dropout = dropout

        #Generate random weights
        self.generateRandomWeights()


    def generateRandomWeights(self):
        def randTuple(x,y, r=5, m=0.1):
            return np.interp(np.round(np.random.randn(x,y),r),(0,1),(-1/math.sqrt(x),1/math.sqrt(x)))
            #return np.round(np.random.randn(x,y), r) * m

        #Generate random weights
        self.W1 = randTuple(self.inputSize, self.hiddenSize[0])
        self.W2 = [
            randTuple(
                self.hiddenSize[i],
                self.hiddenSize[i+1]
            ) for i in range(0, len(self.hiddenSize)-1)]
        self.W2.append(randTuple(self.hiddenSize[-1], self.outputSize))


    def doDropout(self, layer, weight):
        #https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        if weight!=0:
            layer *= np.random.binomial([np.ones((len(layer),len(layer[0])))],1-weight)[0] * (1.0/(1-weight))
        return layer

    def forwardpropagate(self, X):
        #Forward propagate data though the network
        self.z, self.a = [], []
        #From input to hidden layer
        self.z.append( self.doDropout(np.dot(X, self.W1), self.dropout[0]) )
        self.a.append( self.activation(self.z[-1]) )
        #Through the hidden layers
        for i in range(len(self.W2) - 1):
            self.z.append( self.doDropout(np.dot(self.a[-1], self.W2[i]), self.dropout[1]) )
            self.a.append( self.activation(self.z[-1]) )
        #From hidden to output layer
        self.z.append( self.doDropout(np.dot(self.a[-1], self.W2[-1]), self.dropout[2]) )
        self.a.append( self.activation(self.z[-1]) )
        #Return the final value of the output layer
        return self.a[-1]


    def backwardpropagate(self, X, y, yHat):
        #Back propagate the error through the network
        self.error, self.delta  = [], []
        #Back propagate from output to the last hidden layer
        self.error.append( y - yHat )
        self.delta.append( self.error[0]*self.activation(yHat, True) )
        #Back propagate through the hidden layers
        for i in reversed(range(0, len(self.W2))):
            self.error.append( np.dot(self.delta[-1], self.W2[i].T) )
            self.error[-1]
            self.delta.append( self.error[-1]*self.activation(self.a[i], True) )


    def updateWeights(self, X, learnRate):
        #Update the hidden layer weights
        j = 0
        for i in reversed(range(len(self.W2))):
            currDelta = np.dot(self.a[i].T, self.delta[j])
            currDelta -= (self.Lambda*self.W2[i]) #Regularisation - penalise large weight values
            currDelta *= learnRate**0.9 #Learning rate - possible implement momentum

            self.W2[i] += currDelta
            j += 1

        #Update the input layer weights
        currDelta = np.dot(X.T, self.delta[j])
        currDelta -= (self.Lambda*self.W1) #Regularisation - penalise large weight values
        currDelta *= learnRate**0.9 #Learning rate  - possible implement momentum

        #print(sum(sum(currDelta)))

        self.W1 += currDelta

    def getArchitectureString(self):
        output = "A {}-{}-{} fully connected network".format(
            self.inputSize,
            self.hiddenSize,
            self.outputSize,
        )
        return output

    def __repr__(self):
        output = "A {}-{}-{} fully connected network with weights:\n{}\n{}".format(
            self.inputSize,
            self.hiddenSize,
            self.outputSize,
            self.W1,
            self.W2,
        )
        return output

    @staticmethod
    def activation(z, deriv=False):
        #Apply the activation function to scalar, vector, or matrix
        if not deriv:
            return 1/(1+np.exp(-z))
        else:
            return z*(1-z)

    def generateLayers(self):
        self.layers = []
        self.layers.append( Layer(self, self.outputSize, self.W2[-1]) )
        for i in reversed(range(1,len(self.hiddenSize))):
            self.layers.append( Layer(self, self.hiddenSize[i], self.W2[i-1]) )
        self.layers.append( Layer(self, self.hiddenSize[0], self.W1) )
        self.layers.append( Layer(self, self.inputSize, None) )

    def drawNetworkLayout(self, weightMultiplier=20):
        self.generateLayers()
        for layer in self.layers:
            layer.draw(weightMultiplier)
        plt.axis('scaled')
        plt.show()

    def drawHintonDiagram(self):
        diagram = HintonDiagram()
        diagram.show([self.W1] + self.W2)


class Trainer:
    def __init__(self, nn, noEpochs, batchSize, learnRate):
        self.nn = nn
        self.noEpochs = noEpochs
        self.batchSize = batchSize
        self.learnRate = learnRate

    def train(self, X, y, testX, testY, printRate):
        #Train the network
        print(self)

        combinedData = [(X[i].tolist(), y[i].tolist()) for i in range(len(X)-1)]
        self.trainErrors, self.testErrors = [], []

        for epoch in range(self.noEpochs):
            #Do one epoch (a full forward then backward propagation)
            testYHat = self.nn.forwardpropagate(testX)

            if self.batchSize == -1:
                #Train the network on the entire data set
                yHat = self.nn.forwardpropagate(X)
                self.nn.backwardpropagate(X, y, yHat)
                self.nn.updateWeights(X, self.learnRate)

                trainError = self.getMSError(y, yHat)
                testError = self.getMSError(testY, testYHat)

                #print(X[0])
                #print(y[0])
                #print(yHat[0])
                #print(trainError)

            else:
                #Generate minibatches to train the network with
                random.shuffle(combinedData)
                batchX = np.array([d[0] for d in combinedData[:self.batchSize]])
                batchY = np.array([d[1] for d in combinedData[:self.batchSize]])

                yHat = self.nn.forwardpropagate(batchX)
                self.nn.backwardpropagate(batchX, batchY, yHat)
                self.nn.updateWeights(batchX, self.learnRate)

                trainError = self.getNumCorrect(batchY, yHat)
                testError = self.getNumCorrect(testY, testYHat)

            self.trainErrors.append(trainError)
            self.testErrors.append(testError)

            if epoch % printRate==0:
                trainAccuracy = self.getAccuracy(y, yHat)
                print(self.getTrainingString(epoch, trainError, testError, trainAccuracy))


    def getTrainingErrors(self):
        return self.trainErrors, self.testErrors

    def getFinalAccuracy(self, X, y, testX, testY):
        yHat = self.nn.forwardpropagate(X)
        testYHat = self.nn.forwardpropagate(testX)
        return self.getAccuracy(y, yHat), self.getAccuracy(testY, testYHat)

    def getError(self, y, yHat):
        return " ".join([str(self.getMSError(y,yHat)), str(self.getNumCorrect(y,yHat))])

    def getMSError(self, y, yHat):
        return round( np.mean(np.square(y - yHat)), 5 )

    def getAccuracy(self, y, yHat):
        count, total = 0, len(y)
        for i in range(total):
            if np.argmax(y[i]) == np.argmax(yHat[i]):
                count += 1
        return round( count/total*100, 3 )

    def drawMetricGraph(self, metrics, metricName):
        #Plot a graph of epochs against losses
        import matplotlib.pyplot as plt
        for metric in metrics:
            plt.plot(np.arange(0,self.noEpochs), metric)
        plt.xlabel('# epochs')
        plt.ylabel(metricName)
        plt.show()

    def getTrainingString(self, epoch, trainError, testError, trainAccuracy):
        return "@ epoch {}, train error={}, test error={}, train accuracy={}".format(
            epoch, trainError, testError, trainAccuracy
        )

    def getTestingString(self, X, y, testX, testY):
        trainAccuracy, testAccuracy = self.getFinalAccuracy(X, y, testX, testY)
        output = "\nTesting network, after {} epochs\n".format(self.noEpochs)
        output += "\t- Training data accuracy = {}%\n".format(trainAccuracy)
        output += "\t- Testing data accuracy = {}%\n".format(testAccuracy)
        return output

    def __repr__(self):
        output = "Training for {} epochs, with features:\n".format(self.noEpochs)
        output += "\t- Batch size: {}\n".format(self.batchSize)
        output += "\t- Learning rate: {}\n".format(self.learnRate)
        return output


    def playWinner(self, testing=False):
        game = Tictactoe()
        result = game.play(self.nn, testing)
        if result == "Draw":
            print("Draw!")
        else:
            print("{} won!".format(result))


def loadPickleData(filename):
    #Return data extracted from a csv file as two lists, input and expected output
    with open(filename, "rb") as f:
        data = pickle.load(f)
    X = np.array(data[0], dtype=float)
    y = np.array(data[1], dtype=float)
    return X, y


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

    def getEmptySquarePositions(self):
        positions = []
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] == "_":
                    positions.append((x,y))
        return positions

    def aiNeuralMove(self, nn):
        #Use the neural network to generate move
        iL = np.array(self.board, dtype=str).flatten()
        #iL = [{"X": 1, "O": 2, "_": 0}.get(item,item) for item in iL]
        #X = np.nan_to_num(iL/np.amax(iL, axis=0))
        X = np.array([0 if x=="X" else 1 for x in iL]+[0 if x=="O" else 1 for x in iL])
        yHat = nn.forwardpropagate(X)
        print(yHat)
        while True:
            indexMax = np.argmax(yHat)
            move = [indexMax%3, indexMax//3]
            if self.isMoveValid(move):
                return move, X, yHat, indexMax
            else:
                yHat[indexMax] = 0


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
                return 1

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


    def randomMove(self):
        moves = self.getEmptySquarePositions()
        return random.choice(moves)

    def humanMove(self):
        while True:
            move = [int(x) for x in input("Position x,y: ").split(',')]
            if self.isMoveValid(move):
                return move
            else:
                print("Invalid Move")

    def play(self, nn, testing=False):
        while True:
            if 1: #not testing:
                self.displayBoard()
            if not self.playerNum:
                if not testing:
                    print("AI move: ")
                move, *_ = self.aiNeuralMove(nn)
                #print(move)
            else:
                if not testing:
                    print("Player move: ")
                    move = self.humanMove()
                else:
                    move = self.getBestMove()
            self.makeMove(self.players[self.playerNum], move)
            self.playerNum = (self.playerNum+1)%2
            self.winner = self.checkIfWon()
            if self.winner:
                break
        self.displayBoard()
        return self.winner






def main():
    #Initialise the neural network
    inputSize, hiddenSize, outputSize = 18, [30,30], 9
    Lambda, dropout = 0, [0,0,0]
    nn = NeuralNetwork(inputSize, hiddenSize, outputSize, Lambda, dropout)

    #Initialise the trainer
    noEpochs, batchSize, learnRate, printRate = 5000, -1, 0.0005, 500
    myTrainer = Trainer(nn, noEpochs, batchSize, learnRate)

    #Get the training and testing data
    X, y = loadPickleData("train.pickle")
    testX, testY = loadPickleData("test.pickle")

    #Train the network
    myTrainer.train(X, y, testX, testY, printRate)

    #Print the accuracy of the network
    print(myTrainer.getTestingString(X, y, testX, testY))
    #print(nn)

    myTrainer.drawMetricGraph(myTrainer.getTrainingErrors(), "Mean Squared Error")
    nn.drawNetworkLayout()
    nn.drawHintonDiagram()

    while True:
        myTrainer.playWinner()



if __name__ == "__main__":
    exit(main())



"""
Testing network, after 30000 epochs
        - Training data accuracy = 96.948%
        - Testing data accuracy = 99.0%

A 18-[27, 18]-9 fully connected network with weights:
[[ 1.52485852e+00 -1.70041830e+00 -3.12010466e+00 -2.57114374e+00
  -1.34915858e+00 -4.56430785e-01  2.33988683e+00 -9.20732552e-01
   1.23558071e+00 -8.48697059e-01  4.65601733e+00 -7.59437367e-01
   1.80643000e+00  1.97880079e+00  2.61592219e+00 -2.91110693e+00
  -1.57397151e+00 -5.88876776e+00 -3.31044736e+00  5.78935449e+00
   1.36388815e+00  2.58389703e+00  3.72075898e+00  1.34157240e+00
  -2.68071583e+00 -3.94885118e-01 -7.75683762e-01]
 [-3.47048941e+00  1.47212559e+00  1.28242723e+00 -2.03634140e+00
  -7.45034577e-01 -1.72380434e+00 -2.26296406e+00  1.56575898e+00
  -5.40420819e-01 -8.53166952e-01 -1.01109359e+00 -1.23445769e+00
   1.59697950e+00 -1.16925887e+00 -4.88260915e+00 -9.65749641e-01
   1.02305798e+00  7.58856748e-01 -2.98022613e+00  1.72814632e+00
   2.86057639e+00  1.43478078e+00  2.40678507e+00  1.60405202e+00
   1.43946642e+00 -8.07681503e-01  4.00886557e+00]
 [-5.73086295e-02  1.13938991e+00 -3.75773150e+00  1.61110825e+00
  -5.81777213e-01  4.38706901e-01 -1.53001405e+00  2.03108197e+00
  -2.07952576e+00 -1.30682781e+00 -3.55185694e+00 -9.48324631e-01
   2.82245832e+00 -9.82056199e-01  2.28635594e+00  2.96567290e+00
  -3.49017792e+00  1.60784149e+00 -2.62040509e+00 -3.47015676e+00
  -1.79077198e+00 -5.79332636e-01 -1.23676975e+00 -1.64313547e+00
  -9.76258252e-01  7.56260363e-01 -1.25835088e+00]
 [ 1.37200919e+00 -8.90794841e-01 -5.92454925e-01 -2.63154865e+00
   2.26571676e+00  3.44544421e+00  2.75019022e+00 -2.12261309e+00
   2.10396811e+00  1.29618614e+00 -2.22852626e+00 -6.64619382e-01
   3.99612331e-01  1.18311567e+00 -3.17723127e-01 -3.04715788e+00
  -2.67110656e-01  2.73570396e-01  2.81090670e+00  1.32799088e+00
   3.71605677e+00  1.03592573e+00 -1.47066838e-02 -8.74735783e-01
   1.99342697e+00 -1.25804426e+00 -2.60424132e-01]
 [-1.42543648e+00 -5.76148394e-01 -2.37064546e+00  3.52417019e-01
   2.13796438e+00 -1.57331310e+00 -1.96492824e+00 -3.78617696e+00
  -8.89454018e-01 -4.43652509e-01 -2.20907476e+00  6.79400658e-01
  -3.44923654e+00 -3.90783682e+00  1.31311640e-01  2.15821819e+00
   1.81796574e+00 -9.87045251e-01  2.96898879e+00  4.17197322e+00
   2.33709632e-01  2.88518034e+00 -2.84940247e-01 -3.66299912e+00
  -1.96195228e+00  1.64837580e+00  9.24289772e-01]
 [ 1.50777695e+00 -2.01489621e+00 -9.51810409e-01  1.86323089e+00
   2.39034390e+00 -2.80554260e+00  9.80225781e-02  2.81631708e+00
  -2.29509227e+00  4.71668671e-01 -6.52646096e-01 -1.36241814e+00
   1.08154807e+00 -2.02989352e+00  2.23086434e-01 -8.71043303e-01
  -3.22235955e-01  3.08587974e-01  3.36123461e+00  1.33530747e+00
  -1.13521677e+00 -1.35679285e+00 -2.31310083e+00 -1.84187443e+00
   1.51096702e+00 -2.30102556e+00 -6.82771525e-01]
 [ 9.61107815e-01  4.32578288e-01  4.60933981e+00 -5.80172981e-01
  -1.33062654e+00  6.18009564e-01  2.26102977e+00 -3.18206677e+00
   3.01901591e+00 -2.63205319e+00  9.46677008e-01 -1.58773080e+00
  -2.07280310e+00  3.96765696e+00  1.34602141e-01  2.69439013e+00
   6.29766260e+00  2.48288570e-01 -4.99169210e-01 -3.97458198e+00
   2.17093914e+00 -6.66337779e-01 -8.80805526e-01 -5.82291083e-01
   1.05543154e+00  1.11111865e+00 -2.52752202e+00]
 [-2.03576452e+00  2.29603928e+00  1.58566303e+00 -3.15721442e-02
  -1.98212556e+00  1.18784117e+00 -5.81419396e-01 -1.01213410e+00
   1.13060620e+00  1.29800561e+00  8.32349409e-01  6.78448044e-02
  -1.35624574e+00 -4.17057668e-01  3.77770944e-01 -1.07443710e+00
   6.23844369e-01 -3.04673161e-01 -1.42568239e-01  3.76274566e-02
  -4.98984284e+00 -1.98987199e+00  7.27376690e-01  2.20058134e+00
   9.85205456e-01 -1.37709582e+00  3.39304570e+00]
 [ 1.14302619e-02  1.01163105e+00  3.01822445e+00  1.31288293e+00
  -1.99127307e+00 -7.97471910e-02 -9.18018061e-01  3.13958289e-03
  -2.04718797e+00  1.91598093e-01  2.20348923e+00  3.54195922e+00
  -1.25666941e+00  9.06133430e-01  3.23042364e-01 -1.55068212e+00
  -2.10974251e+00  3.24794879e+00 -2.40134652e-01 -2.97622816e+00
   3.31928986e-02 -3.22740023e+00  6.83610068e-01  3.60567796e-01
  -1.96871010e+00 -6.15318591e-01 -3.35938211e+00]
 [-1.43048692e+00  1.50651627e+00 -3.67775328e-01 -4.64562390e-01
  -1.92376505e+00 -3.34605327e-01 -3.82945013e-01  9.67029846e-01
  -3.43248122e+00 -1.98632316e+00  2.04693859e+00  5.02541735e-01
  -2.99611597e+00 -2.54610289e+00  7.16430091e+00  4.16106884e-01
   3.38335524e-02 -3.26465002e+00  1.27900593e+00  2.82311189e+00
  -3.02027178e+00 -1.42009672e+00  3.46038494e+00 -3.00107010e+00
   3.22902727e+00  8.65459063e-01 -1.51115026e+00]
 [ 1.90701225e+00  7.34841315e+00 -3.44586933e+00 -1.60112232e+00
  -7.41934595e-01 -2.89258414e+00  8.93718265e-01 -1.17782003e-01
  -1.57039522e+00  2.13726587e+00 -1.48831468e+00  1.33379162e-01
  -5.88190661e-01  8.16086082e-01  1.88736470e-01  4.35036832e-01
  -1.19911789e+00  5.35719877e-01  1.28168131e+00 -2.96621953e+00
  -3.00729506e-01 -1.93391522e-01  2.54983109e+00 -4.53779245e+00
  -1.60129347e+00 -4.08554789e+00 -1.74810676e+00]
 [ 7.76983423e-02 -4.42153767e+00 -2.64542036e-01  3.49994198e+00
  -2.59604784e+00  3.79100330e-01 -1.21377098e+00 -1.42095398e+00
   3.04350807e+00 -3.68158282e+00 -1.17495868e+00  1.98351325e+00
  -3.06847335e-01  6.07009358e+00 -3.36885334e+00 -2.22049820e+00
   7.89236247e-01  5.89499383e-02  1.92218581e+00  3.19645074e+00
  -6.31047823e-01 -7.89817446e-01 -2.16511327e+00  2.98629256e-01
  -3.77112656e+00  6.86875505e-01 -1.26002951e+00]
 [-3.21126535e+00  7.45572048e-01  1.97713037e+00 -2.36767282e+00
  -1.77556814e+00  4.01019171e+00 -1.67266341e+00  1.99835481e+00
  -3.16621195e+00  2.20770888e+00  1.01303242e-01 -1.01852093e+00
  -2.64118625e+00 -1.57315462e-02 -1.80486526e+00  6.66247438e-01
  -1.35590377e+00 -1.47172091e-01 -2.44228799e+00 -1.75009365e+00
  -1.24831933e+00  8.03983434e-01 -3.09942844e+00  8.00611863e-01
  -3.09367356e+00 -3.22024224e+00  1.62340041e+00]
 [ 3.21716150e+00  5.43658566e-01  1.08804767e+00  2.21195864e+00
  -4.67402840e-01 -2.92455711e+00 -3.53987723e-01 -4.81016448e+00
  -1.58997726e-01 -8.24149082e-01  2.51421105e+00  7.86219364e-01
  -1.78806867e+00  5.60107638e-01  7.83442481e-01 -2.00207073e+00
  -1.83277461e+00  5.63410171e+00 -2.37806692e+00 -3.83744280e+00
  -8.19618009e-01  4.75778210e-02 -1.17422829e-01  5.59374636e+00
   1.81386352e+00 -1.46378513e-01 -5.01110648e-01]
 [-2.48658285e+00  4.12016524e-01  2.38126501e+00 -3.53469655e+00
  -1.33371554e+00 -1.71143675e+00  5.62860708e-01  7.51482402e-01
   3.24819053e+00  3.65879412e-01  2.10160398e-01 -5.55171768e-01
   4.59219898e-01  6.74478826e-01 -1.14913018e+00  2.22379390e+00
   6.37391629e-01  5.93757624e-01 -2.33119020e+00 -4.56682928e-01
   4.47777133e-02  3.91112543e-01 -2.15804811e+00  9.37682609e-01
  -1.41370196e+00  5.39036024e-01  1.44368005e+00]
 [ 1.08112471e+00 -2.97501986e+00  1.31529810e+00  5.83465566e-01
   2.49332919e+00  8.46573472e-01 -2.96163149e+00  2.86073563e+00
  -2.85600634e+00 -2.26350346e+00 -2.37526335e+00 -1.22476638e+00
   2.92540162e+00 -1.77825538e+00  1.28499392e-01 -2.17511784e+00
   2.34016583e+00  2.10678249e-02 -1.13341891e+00  2.13461598e+00
  -1.06255955e+00  7.20266071e-01  2.22556135e+00  1.59314861e+00
  -2.78204849e+00 -2.56173223e-01 -4.74008223e-01]
 [ 2.80302502e+00 -1.45894229e+00 -4.34067340e+00 -4.79899996e-01
   3.00990645e+00  2.25288000e+00  1.52713829e+00  3.11727532e+00
  -1.47137740e+00  4.56406459e-01  2.23381804e-01 -2.15334491e-01
   1.37612899e+00 -7.61144396e-01 -7.78656652e-01 -9.35517995e-01
  -2.86617223e+00 -1.59510988e-01 -9.58268853e-01 -1.50584204e+00
   5.87368711e-01  2.29349554e-01 -1.21407414e+00 -2.25970840e+00
  -1.34831778e+00  7.05443474e-01  1.79290408e-01]
 [-3.00141163e+00 -2.61244885e+00 -5.52709589e-01  8.55864902e-01
   1.46573097e+00 -2.39436976e-01 -9.63034070e-01  3.24074805e-01
   2.35940134e+00  4.45439941e+00  2.16233176e-01  3.89231845e-01
   9.09561733e-01  5.03619502e-02 -1.79962032e+00 -3.69834636e-01
   7.90366040e-01 -1.80084959e+00 -5.00680852e-01  3.31563444e+00
   1.77835480e+00  3.13289494e+00  7.29766685e-01 -2.73431149e+00
   4.05328193e+00  5.13172245e+00  3.10999271e-02]]
[array([[ 8.96510368e-01,  1.60774990e+00,  2.85580199e+00,
        -3.91354796e+00,  1.34996443e+00, -3.94331586e-01,
         5.29557587e+00,  9.33411953e-01, -4.20367621e-02,
         1.28817278e+00, -6.95107167e-01, -6.70921025e-01,
        -1.97855476e+00,  2.53651672e+00, -2.91523263e+00,
         3.56224881e+00, -5.65813614e-01, -3.87860854e-01],
       [ 1.92483204e+00, -2.28451557e+00, -3.76829229e+00,
        -2.40739494e+00,  6.94888154e-01, -8.69796671e-01,
        -2.69824044e-01,  2.78299977e+00, -1.40464777e+00,
         1.09097574e+00, -2.00467747e+00,  1.81288740e+00,
        -2.56620794e+00,  9.96755591e-01, -3.27183044e+00,
         1.14887777e+00,  5.96783791e-01, -2.13035419e+00],
       [-1.05194335e+00,  2.59053782e+00,  1.58436801e+00,
         8.90735705e-01,  2.26722827e+00,  2.68052821e+00,
         1.78343848e+00, -3.58698901e-01, -3.38982649e+00,
        -7.12773339e-01, -2.49327987e+00, -2.28611533e+00,
        -2.34153506e+00,  1.81770018e+00, -2.15267003e+00,
        -4.41638811e+00,  1.36351488e+00,  1.82448384e+00],
       [ 3.38679956e+00,  1.87074668e-01, -1.92013928e+00,
         7.83352680e-01,  2.48698162e+00, -3.58172943e+00,
        -5.12130314e+00,  1.24062249e-01, -3.38888095e-01,
         3.17888907e-01,  2.47615204e+00,  8.90205783e-01,
        -4.26067332e-01, -1.20111825e+00, -3.11359149e+00,
         3.74258522e+00, -6.26266043e-01, -1.55482863e+00],
       [-2.63174345e+00,  2.12729200e+00,  2.31253575e+00,
         1.10065499e+00,  1.38405556e+00,  1.33080627e+00,
        -1.03964326e+00, -1.17993006e+00, -3.82536192e+00,
        -4.06465516e+00, -2.67468566e+00, -2.76577107e+00,
        -1.58338706e+00, -6.70450781e-01, -3.02165706e+00,
         4.85118259e-01, -2.57236582e+00,  5.17878378e+00],
       [ 2.70219327e+00, -1.04570268e+00, -1.25514139e+00,
         1.41030711e+00, -2.76104748e+00, -9.10492417e-01,
        -3.89648280e+00,  7.18404898e-01,  3.15933849e+00,
        -4.39048609e+00,  1.98395658e+00,  3.34297617e-01,
        -4.34200553e+00,  4.57447509e-01, -3.70938461e-01,
         4.80974810e-01,  9.90180171e-01,  9.15702630e-01],
       [-1.43928550e+00,  1.80858655e+00,  2.58724344e+00,
        -2.33939533e+00, -8.08642797e-01,  2.82149477e+00,
        -3.32945311e+00,  2.28231860e+00, -8.02333587e-01,
        -6.96487944e-01, -1.89377762e-01, -2.54199181e+00,
        -2.22464146e+00, -9.98042042e-01,  2.20892888e+00,
        -3.82188359e+00,  2.99010570e+00,  4.66425876e+00],
       [ 8.10720739e-01, -1.13723363e-01, -4.38541431e+00,
         1.88613184e-01,  3.89783329e+00,  2.52964284e+00,
        -3.94544759e+00, -2.00447258e+00, -2.97160291e+00,
        -1.55810113e+00,  1.84393062e+00,  2.14388370e+00,
        -8.21893311e-01,  1.85773200e+00, -7.41009184e-01,
        -5.83077258e+00,  2.50716739e+00, -9.96837327e-01],
       [ 1.61937459e+00, -1.36263640e+00, -2.66575018e+00,
         1.68618765e+00,  4.18193475e+00,  2.87008000e-01,
        -5.46934029e+00, -3.52541476e+00, -2.05101983e+00,
        -4.02373765e+00,  2.73746116e+00,  2.18202342e+00,
         3.09361950e+00, -1.39394656e+00, -1.27352865e-01,
        -3.16027232e+00,  1.48528681e+00, -3.33081916e+00],
       [ 1.59361980e+00, -1.04308089e+00, -2.65025878e-01,
         2.84010672e+00,  2.49104699e+00, -6.22061340e-03,
         6.70334164e-01, -5.71824706e-01,  6.98278261e-02,
        -1.13789632e+00, -2.18686259e+00,  9.28514506e-01,
        -3.10486140e+00,  6.75550565e-01, -5.59600801e+00,
        -1.25608303e+00,  2.45326426e+00, -1.14406777e+00],
       [-5.40104145e+00,  9.02233759e-01,  3.16257130e-02,
         1.52742739e+00,  3.79683161e-01,  3.15742397e+00,
        -1.01292790e+00,  1.19401138e+00, -3.53155493e+00,
        -1.54150535e+00,  9.69591771e-01, -2.91715048e+00,
         1.24909414e+00,  1.28345298e-02,  6.61536550e+00,
        -2.93005700e+00, -1.99602738e+00,  1.70437501e+00],
       [ 1.02244534e+00, -2.46058887e+00,  6.83930381e-01,
         6.49102862e+00,  4.97185536e-01,  1.37825733e+00,
        -9.85309848e-01,  1.61297126e+00, -4.48772404e-01,
         8.30245035e-01, -1.90685179e+00, -2.89071281e+00,
         1.12994627e+00,  2.04374925e+00, -1.43808417e+00,
        -6.76770580e-01, -2.09079672e+00,  8.40434935e-01],
       [ 2.73654468e+00,  1.32222908e+00, -4.82264448e+00,
         1.03235992e-01, -3.04798845e-01,  1.07770466e+00,
        -2.49620967e+00,  8.21292685e-01, -2.85622403e+00,
        -3.39566559e-01,  2.82385980e-01, -1.52923113e+00,
         1.31176531e+00,  2.80390824e+00, -3.73265231e+00,
         5.59739825e-01,  1.65106447e-01, -1.36281748e+00],
       [ 1.04109326e+00, -3.79389286e+00, -4.00139467e+00,
        -2.54167126e+00, -3.12246639e+00, -3.12345467e+00,
        -1.30824953e+00,  1.31854354e-01, -4.21047625e-01,
         1.60856285e+00,  9.12315165e-01,  1.14339640e+00,
         2.64230281e+00, -1.07120726e+00, -2.71378783e+00,
         4.13498701e+00,  3.50704808e-01, -1.26965094e+00],
       [-3.12650836e+00, -1.10240064e-02, -2.75387051e+00,
         1.50498536e+00, -1.34656085e+00,  2.14208709e-01,
        -5.88841640e+00, -3.97141051e+00,  1.74743331e+00,
        -1.14114527e+00,  3.94203486e+00, -5.46144739e-01,
         2.42353998e+00, -2.30284257e+00,  6.09069045e+00,
        -6.74402198e-01, -1.81855116e+00,  1.47246414e+00],
       [-1.41121919e-01,  2.63421561e+00,  1.41581709e+00,
        -3.84842991e+00, -4.69190097e-01, -1.58660044e+00,
         3.56763352e+00,  1.42559522e+00,  1.03733001e+00,
         2.38049608e+00, -1.71767870e+00,  1.01894195e+00,
         4.55219041e+00, -2.54808526e+00,  6.80151066e-01,
         5.59616673e+00, -1.70611237e+00,  7.55474725e-01],
       [ 4.06326808e+00,  5.37218778e+00, -9.31573949e-01,
         2.61484471e-01,  3.59518220e+00,  1.17545968e+00,
        -5.65674384e-01,  2.74244726e+00, -1.86986636e+00,
        -1.32418828e+00, -1.83761211e+00, -8.62655946e-01,
        -1.69529047e+00, -2.28506134e+00, -4.61791630e+00,
         1.11381893e-01, -1.21034170e+00,  6.01559213e-01],
       [-3.41111240e-01,  4.49643828e-01,  2.20205464e+00,
        -2.34387995e+00, -1.48632341e+00, -1.62805929e+00,
         2.10308207e+00, -1.98386166e+00,  3.60216844e+00,
         8.31808530e-01, -1.14871214e+00,  1.70653165e+00,
        -7.94085911e-01, -1.45030686e+00, -6.87796649e+00,
         3.73002717e+00,  2.12010420e+00, -7.15290045e-03],
       [-4.87993023e+00, -2.37956263e+00,  1.29461409e+00,
        -5.22688361e+00,  1.37733569e+00,  4.10645992e-01,
         1.73230678e+00, -4.08493277e+00,  4.05183963e+00,
         4.64349189e+00,  9.83075134e-01,  2.33816392e+00,
         2.13196366e+00,  1.74280662e+00, -8.39154457e-01,
         6.63665884e-01,  4.74866748e+00, -3.73311075e-02],
       [ 2.53822244e-01, -6.52549156e-01, -2.60849874e+00,
        -2.99262955e+00, -3.40846607e+00, -2.16179522e+00,
        -3.78128179e+00,  1.67030089e+00,  1.07813484e+00,
        -7.67799580e-01,  3.92224473e+00,  1.28448562e+00,
         1.59406312e+00, -1.34385223e+00,  3.38397344e+00,
        -1.02315787e+00, -1.44121213e+00, -2.65044042e+00],
       [-7.83332859e-02, -1.09321209e-01, -2.56046829e+00,
         2.23017660e+00, -1.25400883e+00,  2.43611011e+00,
         2.44495001e-01,  9.53451370e-02, -1.96316631e-01,
         1.75618800e+00, -1.43132514e+00,  4.81254993e+00,
        -2.88865024e+00,  1.58084172e+00,  2.45766867e+00,
        -4.84128651e+00,  1.40217961e+00, -2.77723628e+00],
       [-3.16819110e+00, -2.79318847e+00,  8.41178019e+00,
         1.78465125e+00, -2.50563296e+00,  3.84430244e-01,
         5.32030106e+00,  8.70638232e-01,  4.10868160e+00,
         7.73210731e-01, -4.27522498e+00, -1.85908379e+00,
         2.31907926e+00, -2.44160659e+00, -4.56912963e-01,
         1.13360301e+00, -2.62267428e+00,  2.98306284e+00],
       [-2.53239931e-02,  5.68879412e-01, -1.20864260e+00,
        -1.54172745e+00, -1.64872016e+00,  1.44755473e+00,
         3.67636560e+00,  3.02129582e+00, -3.33519847e+00,
        -5.67755821e-01, -1.81775107e+00,  9.94086215e-01,
        -5.38363776e-01,  5.50562621e-01,  3.68091843e+00,
        -4.51544300e+00, -9.27564799e-01,  3.08851510e-01],
       [ 3.20221289e-01,  2.18263311e-01,  2.10687254e+00,
        -2.38830452e+00, -4.58178937e-01, -1.19397973e+00,
        -3.53286605e-01, -3.27203683e+00,  5.01053205e+00,
         1.15264305e+00,  1.40606159e-01,  2.06065017e+00,
         1.94467049e+00, -4.45494964e+00, -8.74742819e-01,
         2.65436691e+00, -1.71812291e+00,  2.16537722e+00],
       [-7.22922789e-01, -3.18330057e+00, -1.03244659e+00,
         2.81459866e+00, -2.05574010e+00,  1.87222142e+00,
         1.88923598e+00,  1.03196764e+00, -4.08291856e-01,
         5.93401652e-01, -9.10828749e-01,  1.81467514e-01,
        -1.35633982e-01,  5.02238860e+00,  4.63750503e+00,
        -3.88508156e+00, -3.84532299e+00,  1.63371595e+00],
       [ 2.78747563e+00,  2.44703985e-01,  6.27129820e-01,
        -2.82959304e+00,  2.11402339e+00, -4.34619448e+00,
        -1.52784099e+00, -3.14004028e+00,  8.76540173e-01,
         1.51513602e+00,  2.22225141e+00, -1.04781198e+00,
         1.49685664e+00,  6.89455285e-03, -5.07722481e+00,
         5.37010690e+00, -7.08383202e-01, -5.74878925e-01],
       [-1.89550862e+00, -3.60555119e+00, -4.85991061e-01,
        -4.69079805e+00, -9.89117390e-01,  1.11855207e+00,
         3.46081315e+00,  6.77972124e-01,  2.41822320e-02,
         3.13818865e+00,  4.69440383e-01, -1.34918460e+00,
        -1.50255806e+00,  3.11999794e+00,  1.91616527e-02,
         1.92437179e+00,  3.68467563e+00,  4.28105046e-01]]), array([[-4.42372178, -1.12330126,  3.97470911, -2.67855229, -6.78128103,
        -6.59602805,  4.76209957,  0.48392016,  2.43354727],
       [-1.45849869, -2.01721118, -5.4729799 , -1.54224388, -2.54482478,
         0.69254412,  8.06520634, -4.69495752, -4.642246  ],
       [-5.18630029, -4.72813776, -7.58099941,  3.7135203 ,  8.28705737,
        -3.6918055 , -4.70187633, -3.3721709 , -2.62020782],
       [-3.93254545, -6.40404665, -4.2790507 ,  1.70294483, -1.89930243,
        -5.59937352, -4.20582476, -5.70928578,  9.40895241],
       [-7.39260153, -2.50777003, -0.89340461, -5.60807686, -1.40856068,
         6.03130951,  0.56723201, -0.30138038,  3.27099908],
       [-2.94909493, -3.17101905, -8.6858613 ,  1.3361476 , -3.81475881,
         2.01504638, -1.97341874, -0.04753039,  1.61125676],
       [-6.54246594,  8.7853735 , -8.31188962, -4.22620313,  4.45051031,
        -4.15135814, -2.57692543, -3.5326778 , -2.39790092],
       [ 1.89670148,  1.469103  , -5.64591907, -4.26512001, -2.47640572,
        -5.52379469,  4.16870346,  2.93270089, -2.78383628],
       [-3.47256814, -5.72637259,  4.54979573,  6.75905349, -0.02061888,
        -5.9638949 , -4.44321657, -3.0921542 , -4.76902507],
       [-4.91668148,  4.17634931,  1.0895943 , -3.45009421,  1.47786888,
        -1.91002112, -5.19841185, -0.2706254 , -4.33205476],
       [ 2.2156737 , -5.15461324,  3.89632733, -4.79933526, -8.79722236,
        -1.9544658 , -4.14335476,  0.99924291, -3.20519783],
       [ 1.40331287,  5.21754188,  1.35380816, -0.13857817, -5.99664659,
         1.54446187, -2.72442048, -9.57960583, -1.02749862],
       [ 4.72974883, -4.79143504, -2.45521148, -5.83578685,  3.34311601,
         3.31596453, -6.13339563, -5.1253376 , -0.90611362],
       [ 1.08456038,  3.46873376, -1.33707647, -4.37041663, -2.46984448,
        -0.34006896, -7.59681374,  1.73803504,  1.86780841],
       [ 9.47584229, -4.94337482, -5.39092345, -6.49073911, -3.56778772,
        -6.28342243, -5.31379407, -6.40201165, -7.33033079],
       [-9.16958432, -6.30136606,  6.16325401, -7.56988864,  5.31723294,
        -2.16651416,  0.02582215,  2.03497535, -0.71935311],
       [-2.17387276,  0.22587565, -0.35192851,  4.40780603, -7.12153439,
         2.5394809 , -3.81862393,  2.42995432, -7.30745519],
       [ 0.12627477, -7.29518344, -6.26099494,  3.11541507,  3.16004846,
        -2.222997  ,  1.73509535,  0.8191878 , -3.72490124]])]
"""
