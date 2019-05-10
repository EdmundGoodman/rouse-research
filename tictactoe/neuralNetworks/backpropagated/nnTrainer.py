#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5
import os, random, copy, math, time, itertools, pickle
import numpy as np

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk as itk


class TictactoeBoard:
    def __init__(self):
        #Set the game variables
        self.players = ["O", "X"]
        self.resetGameVariables()

    def resetGameVariables(self):
        #Reset the board, the current player, and the winner
        self.board = [["_" for _ in range(3)] for _ in range(3)]
        self.playerNum = 0

    def getBoard(self):
        return self.board

    def setBoard(self, board):
        self.board = copy.deepcopy(board)

    def getPlayerNum(self):
        return self.playerNum

    def setPlayerNum(self, playerNum):
        self.playerNum = playerNum

    def togglePlayer(self):
        self.playerNum = (self.playerNum+1)%2

    def getPlayers(self):
        return [self.getCurrentPlayer(), self.getNextPlayer()]

    def getCurrentPlayer(self):
        return self.players[self.playerNum]

    def getNextPlayer(self):
        return self.players[(self.playerNum+1)%2]

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

    def makeMove(self, move):
        #Make the move
        self.board[move[1]][move[0]] = self.getCurrentPlayer()

    def isMoveValid(self, position):
        #Check if the move is valid
        if 0<=position[0]<=2 and 0<=position[1]<=2:
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

    def isPlayableBoard(self):
        if self.checkIfWon():
            return False

        flatBoard = flattenArray(self.board)
        Os, Xs = flatBoard.count("O"),  flatBoard.count("X")
        #if Os != Xs-1 and Os != Xs and Os != Xs+1:
        if Os != Xs-1: #and Os != Xs:
            return False

        return True


    def __repr__(self):
        output = ""
        for y in self.board:
            output += ",".join(y)+"\n"
        return output


class Tictactoe:
    def __init__(self):
        self.board = TictactoeBoard()

    def setBoard(self, newBoard):
        self.board.board = newBoard

    def minimax(self, board, players, utilities=[1,-1,0,0], alpha=-math.inf, beta=math.inf, leafDepth=9, depth=0):

        if depth>leafDepth: #Simulate leaf nodes beyond recursion depth
            return utilities[3]

        if depth==0:
            maxValue, bestMove = -math.inf, None
            for move in board.getEmptySquarePositions():
                nextBoard = TictactoeBoard()
                nextBoard.setBoard(board.getBoard())
                nextBoard.setPlayerNum(board.getPlayerNum())
                nextBoard.makeMove(move)
                nextBoard.togglePlayer()

                value = self.minimax(nextBoard, players, utilities, alpha, beta, leafDepth, depth+1)
                if value > maxValue:
                    bestMove, maxValue = move, value

            return bestMove


        winner = board.checkIfWon()
        if winner is not False:
            if winner == players[0]: #Win
                return utilities[0]
            elif winner == players[1]: #Loss
                return utilities[1]
            else: #Draw
                return utilities[2]

        if board.getCurrentPlayer() == players[0]: #Maximising layer
            value = -math.inf
            for move in board.getEmptySquarePositions():
                nextBoard = TictactoeBoard()
                nextBoard.setBoard(board.getBoard())
                nextBoard.setPlayerNum(board.getPlayerNum())
                nextBoard.makeMove(move)
                nextBoard.togglePlayer()

                #print(nextBoard)

                func = self.minimax(nextBoard, players, utilities, alpha, beta, leafDepth, depth+1)

                value = max(value, func)
                alpha = max(alpha, value)
                if depth==0: print(move, alpha, beta, 1)
                if alpha >= beta:
                    if depth==0: print("Hit 1")
                    break

        else:
            value = math.inf
            for move in board.getEmptySquarePositions():
                nextBoard = TictactoeBoard()
                nextBoard.setBoard(board.getBoard())
                nextBoard.setPlayerNum(board.getPlayerNum())
                nextBoard.makeMove(move)
                nextBoard.togglePlayer()

                func = self.minimax(nextBoard, players, utilities, alpha, beta, leafDepth, depth+1)

                value = min(value, func)
                beta = min(beta, value)
                #if depth==1: print(move, alpha, beta, 2)
                if alpha >= beta:
                    #if depth==1: print("Hit 2")
                    break

        if depth == 0:
            return move
        else:
            return value


    def getBestMove(self):
        duplicateBoard = TictactoeBoard()
        duplicateBoard.setBoard(self.board.getBoard())
        duplicateBoard.setPlayerNum(self.board.getPlayerNum())
        #x = time.time()
        val = self.minimax(duplicateBoard, duplicateBoard.getPlayers())
        #print(val, round(time.time()-x, 5))
        return val

def flattenArray(arr):
    return list(np.array(arr).flatten())

def unflattenArray(arr):
    return list(np.array(arr).reshape((3,3)))

#Initialise the game
game = Tictactoe()
possItems = itertools.product("OX_", repeat=9)

X,y = [], []
count = 0 #19683 total, 2097 playable for Os

for item in possItems:
    game.setBoard(unflattenArray(item))
    if game.board.isPlayableBoard():
        inputData = [0 if x=="X" else 1 for x in item]+[0 if x=="O" else 1 for x in item]
        outputData = [0]*9
        val = game.getBestMove()
        index = val[1]*3+val[0]
        outputData[index] = 10

        X.append(inputData)
        y.append(outputData)

        if item == tuple("OOXOX_X__"):
            print(game.board)
            print(val)

        """if count < 5:
            print(item)
            print(game.board)
            print(val)"""

        count += 1

print(count)

with open('train2.pickle', 'wb') as f:
    pickle.dump([X, y], f)

combinedData = [(X[i], y[i]) for i in range(len(X)-1)]
random.shuffle(combinedData)
testX = [d[0] for d in combinedData[:100]]
testY = [d[1] for d in combinedData[:100]]

with open('test2.pickle', 'wb') as f:
    pickle.dump([testX, testY], f)
