#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5
import numpy as np
import os, random, pickle, math, copy
from visualisation import *

random.seed(1)
np.random.seed(1)

"""
A first principles neural network with the features:
 - Multiple hidden layers (DNN)
 - Weight picking
 - Forward propagation
 - Back propagation
 - Hinton's dropout
 - Regularisation using a Lambda factor
 - Learning rates
 - Batch & mini-batch data sets

The networks can be displayed using:
 - Hinton diagrams
 - Network architecture diagrams
 """

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
        """Forward propagate an input vector X through the network composed of
        the weights self.W1 and self.W2, storing intermediary variables self.z
        and self.a to allow back propagation, and return the output vector
        of the process
        """

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
        """Back propagate the error of the most recent forward propagation
        through the network. producing the arrays self.error and self.delta
        to allow weight updating
        """

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
        """Update the weights of the network based on the previous pass of back
        propagation in order to improve the network model
        """

        #Update the hidden layer weights
        j = 0
        for i in reversed(range(len(self.W2))):
            currDelta = np.dot(self.a[i].T, self.delta[j])
            currDelta -= (self.Lambda*self.W2[i]) #Regularisation - penalise large weight values
            currDelta *= learnRate #Learning rate - possible implement momentum

            self.W2[i] += currDelta
            j += 1

        #Update the input layer weights
        currDelta = np.dot(X.T, self.delta[j])
        currDelta -= (self.Lambda*self.W1) #Regularisation - penalise large weight values
        currDelta *= learnRate #Learning rate  - possible implement momentum

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

    def drawNetworkLayout(self, weightMultiplier=1):
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
            else:
                #Generate minibatches to train the network with
                combinedData = [(X[i].tolist(), y[i].tolist()) for i in range(len(X)-1)]
                random.shuffle(combinedData)
                batchX = np.array([d[0] for d in combinedData[:self.batchSize]])
                batchY = np.array([d[1] for d in combinedData[:self.batchSize]])

                yHat = self.nn.forwardpropagate(batchX)
                self.nn.backwardpropagate(batchX, batchY, yHat)
                self.nn.updateWeights(batchX, self.learnRate)

                trainError = self.getMSError(batchY, yHat)
                testError = self.getMSError(testY, testYHat)

            self.trainErrors.append(trainError)
            self.testErrors.append(testError)

            if epoch % printRate==0:
                print(self.getTrainingString(epoch, trainError, testError))


    def getTrainingErrors(self):
        return self.trainErrors, self.testErrors

    def getFinalAccuracy(self, X, y, testX, testY):
        yHat = self.nn.forwardpropagate(X)
        testYHat = self.nn.forwardpropagate(testX)
        return self.getAccuracy(y, yHat), self.getAccuracy(testY, testYHat)

    def getMSError(self, y, yHat):
        return round( np.mean(np.square(y - yHat)), 5 )

    def getAccuracy(self, y, yHat):
        #Return the percentage of correct classifications of X
        noCorrect = 0
        yHat = yHat.tolist()
        for rowNum in range(len(yHat)):
            maxOIndex = yHat[rowNum].index(max(yHat[rowNum]))
            if y[rowNum][maxOIndex]==1:
                noCorrect += 1
        percentCorrect = round( (noCorrect/len(yHat))*100, 3 )
        return percentCorrect

    def drawMetricGraph(self, metrics, metricName):
        #Plot a graph of epochs against losses
        import matplotlib.pyplot as plt
        for metric in metrics:
            plt.plot(np.arange(0,self.noEpochs), metric)
        plt.xlabel('# epochs')
        plt.ylabel(metricName)
        plt.show()

    def getTrainingString(self, epoch, trainError, testError):
        return "@ epoch {}, train error={}, test error={}".format(
            epoch, trainError, testError
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

def loadPickleData(filename):
    #Return data extracted from a csv file as two lists, input and expected output
    with open(filename, "rb") as f:
        data = pickle.load(f)
    X = np.array(data[0], dtype=float)
    y = np.array(data[1], dtype=float)
    return X, y


def main():
    #Initialise the neural network
    inputSize, hiddenSize, outputSize = 4, [3], 3
    Lambda, dropout = 0.05, [0,0.2,0]
    nn = NeuralNetwork(inputSize, hiddenSize, outputSize, Lambda, dropout)

    #Initialise the trainer
    noEpochs, batchSize, learnRate, printRate = 100000, -1, 0.0005, 5000
    myTrainer = Trainer(nn, noEpochs, batchSize, learnRate)

    #Get the training and testing data
    X, y = loadPickleData("irisTrainData.pickle")
    testX, testY = loadPickleData("irisTestData.pickle")

    #Train the network
    myTrainer.train(X, y, testX, testY, printRate)

    #Print the accuracy of the network
    print(myTrainer.getTestingString(X, y, testX, testY))
    print(nn)

    myTrainer.drawMetricGraph(myTrainer.getTrainingErrors(), "Mean Squared Error")
    nn.drawNetworkLayout()
    nn.drawHintonDiagram()



if __name__ == "__main__":
    exit(main())
