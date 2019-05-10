import matplotlib.pyplot as plt
from math import cos, sin, atan
import numpy as np
import pylab

class Neuron():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.radius = r

    def draw(self):
        #Flipped horizontally
        circle = plt.Circle((self.y, self.x), radius=self.radius, fill=False)
        plt.gca().add_patch(circle)


class Layer():
    #https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
    def __init__(self, network, numNeurons, weights):
        self.layerDistance = 6
        self.neuronDistance = 2
        self.neuronRadius = 0.5
        self.maxLayerWidth = 4

        self.prevLayer = self.getPrevLayer(network)
        self.y = self.getYPosition()
        self.neurons = self.intialiseNeurons(numNeurons)
        self.weights = weights

    def intialiseNeurons(self, numNeurons):
        neurons = []
        x = self.getLeftMargin(numNeurons)
        for iteration in range(numNeurons):
            neuron = Neuron(x, self.y, self.neuronRadius)
            neurons.append(neuron)
            x += self.neuronDistance
        return neurons

    def getLeftMargin(self, numNeurons):
        return self.neuronDistance * (self.maxLayerWidth - numNeurons) / 2

    def getYPosition(self):
        if self.prevLayer:
            return self.prevLayer.y + self.layerDistance
        else:
            return 0

    def getPrevLayer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def drawWeightLine(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        xOffset = self.neuronRadius * sin(angle)
        yOffset = self.neuronRadius * cos(angle)
        xData = (neuron1.x - xOffset, neuron2.x + xOffset)
        yData = (neuron1.y - yOffset, neuron2.y + yOffset)

        linewidth = abs(linewidth/100)

        #Flipped lines to be horizontal
        line = plt.Line2D(yData, xData, linewidth=linewidth)
        plt.gca().add_line(line)

    def draw(self, weightMultiplier=200):
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron.draw()
            if self.prevLayer:
                for j in range(len(self.prevLayer.neurons)):
                    prevLayerNeuron = self.prevLayer.neurons[j]
                    weight = self.prevLayer.weights[i, j]
                    self.drawWeightLine(neuron, prevLayerNeuron, weight*weightMultiplier)


class HintonDiagram:
    #https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/HintonDiagrams.html
    def _blob(self,x,y,area,colour):
        hs = np.sqrt(area) / 2
        xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
        ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
        pylab.fill(xcorners, ycorners, colour, edgecolor=colour)

    def show(self,Ws):
        reenable = False
        if pylab.isinteractive():
            pylab.ioff()
        pylab.clf()
        pylab.axis('off')
        pylab.axis('equal')

        offset = 0
        for W in Ws:

            height, width = W.shape
            weight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2)) #np.max(W)

            pylab.fill(np.array([0+offset,width+offset,width+offset,0+offset])
                        ,np.array([0,0,height,height]),'gray')

            for x in range(width):
                for y in range(height):
                    _x = x+1
                    _y = y+1
                    try:
                        w = W[y,x]

                        if w > 0:
                            self._blob(_x - 0.5 + offset, height - _y + 0.5, min(1,w/weight),'white')
                        elif w < 0:
                            self._blob(_x - 0.5 + offset, height - _y + 0.5, min(1,-w/weight),'black')
                    except IndexError:
                        pass

            offset += width + 1

        if reenable:
            pylab.ion()
        pylab.show()
