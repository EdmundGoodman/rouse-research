import pickle, math, random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
import numpy as np

#Finished

def loadPickleData(filename, batchSize=-1):
    #Return data extracted from a csv file as two lists, input and expected output
    with open(filename, "rb") as f:
        data = pickle.load(f)
    batchSize = min([batchSize, len(data[0])])
    X = data[0][:batchSize]
    y = data[1][:batchSize]
    return X, y

def plotIrisGraph(X,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    wSeq = [i[0] for i in X]
    xSeq = [i[1] for i in X]
    ySeq = [i[2] for i in X]
    zSeq = [i[3]*20 for i in X]

    #Plot a 3d graph, with the 3 axis and marker size delimiting variables, and colour showing group type
    ax.scatter(xSeq,ySeq,zSeq, s=zSeq, c=y)
    plt.show()

X, y = loadPickleData("irisTrainData.pickle")
plotIrisGraph(X,y)

testX, testY = loadPickleData("irisTestData.pickle")
plotIrisGraph(testX,testY)
