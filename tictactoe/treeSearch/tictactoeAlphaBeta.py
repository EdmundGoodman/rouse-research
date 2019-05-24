#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5
import os, random, copy, math, time

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

    def __repr__(self):
        output = ""
        for y in self.board:
            output += ",".join(y)+"\n"
        return output


class Tictactoe:
    def __init__(self):
        self.board = TictactoeBoard()

        self.root = tk.Tk()
        self.imageSizes = (200,200)
        self.images = self.getImages()
        self.grid = self.initialiseBoard()

        self.enemy = None

    def getImages(self):
        width, height = self.imageSizes[0], self.imageSizes[1]
        image = Image.open('none.jpg').resize((height, width), Image.ANTIALIAS)
        none = itk.PhotoImage(image)
        image = Image.open('o.jpg').resize((height, width), Image.ANTIALIAS)
        o = itk.PhotoImage(image)
        image = Image.open('x.jpg').resize((height, width), Image.ANTIALIAS)
        x = itk.PhotoImage(image)
        return (none, o, x)

    def initialiseBoard(self):
        grid = []
        for i,row in enumerate(self.board.getBoard()):
            gridRow = []
            for j,column in enumerate(row):
                lbl = tk.Label(self.root,image=self.images[0])
                lbl.grid(row=i, column=j)
                lbl.bind('<Button-1>',lambda e,i=i,j=j: self.onClick(i,j,e))
                gridRow.append(lbl)
            grid.append(gridRow)
        return grid

    def onClick(self,i,j,event):
        player = self.board.getCurrentPlayer()
        image = self.images[1] if player=="O" or self.enemy=="ai" else self.images[2]

        if not self.doHumanMove((j,i),event,image):
            return None

        self.board.togglePlayer()

        if self.enemy == "ai":
            self.doAIMove(self.images[2])
            self.board.togglePlayer()

    def doHumanMove(self,move,event,image):
        if self.board.isMoveValid(move):
            self.board.makeMove(move)
            event.widget.config(image=image)
            winner = self.board.checkIfWon()
            if winner:
                self.endGame(winner)
            return True
        else:
            return False

    def doAIMove(self,image):
        move = self.getBestMove()
        self.board.makeMove(move)
        self.grid[move[1]][move[0]].config(image=image)
        winner = self.board.checkIfWon()
        if winner:
            self.endGame(winner)

    def endGame(self, winner):
        #print(self.board)
        message = "Draw!" if winner=="Draw" else 'Player {} won!'.format(winner)
        messagebox.showinfo('End of play', message)
        self.root.destroy()
        exit()

    def playHuman(self):
        self.enemy = "human"
        self.root.mainloop()

    def playAI(self):
        self.enemy = "ai"
        self.root.mainloop()




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
        x = time.time()
        val = self.minimax(duplicateBoard, duplicateBoard.getPlayers())
        print(val, round(time.time()-x, 5))
        #print(val)
        return val

#Initialise the game
game = Tictactoe()
game.playAI()


"""
(1, 1) 2.74894
(0, 2) 0.04987
(1, 0) 0.0027
(2, 1) 0.00037
"""
