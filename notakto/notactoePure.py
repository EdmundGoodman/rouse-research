#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5
#https://arxiv.org/pdf/1301.1672v1.pdf

import os, random, copy

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk as itk

from data import *

#Implement private variables

#Possibly working - tested against insane mode on notakto, (hopefully the perfect model)

class NotactoeBoard:
    def __init__(self):
        #Set the game variables
        self.player = "X"
        self.resetGameVariables()

    def resetGameVariables(self):
        #Reset the board, the current player, and the winner
        self.board = [[["_" for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.playerNum = 0
        self.winner = False

    def checkIfBoardComplete(self, index):
        #Return the winner (Name/Draw/False)
        b = self.board[index]
        p = self.player
        #All the win conditions for tic tac toe
        if (b[2][0] == p and b[2][1] == p and b[2][2] == p): return p
        elif (b[1][0] == p and b[1][1] == p and b[1][2] == p): return p
        elif (b[0][0] == p and b[0][1] == p and b[0][2] == p): return p
        elif (b[2][0] == p and b[1][0] == p and b[0][0] == p): return p
        elif (b[2][1] == p and b[1][1] == p and b[0][1] == p): return p
        elif (b[2][2] == p and b[1][2] == p and b[0][2] == p): return p
        elif (b[2][0] == p and b[1][1] == p and b[0][2] == p): return p
        elif (b[2][2] == p and b[1][1] == p and b[0][0] == p): return p
        if set([x for sublist in b for x in sublist]).isdisjoint(set("_")):
            return "Draw"
        return False

    def checkIfWon(self):
        flag = True
        for i in range(len(self.board)):
            if not self.checkIfBoardComplete(i):
                flag = False
        if flag:
            return str( (self.playerNum+1)%2 )
        else:
            return False

    def getAvailableBoards(self):
        #Return a list containing the indices of available boards
        return [i for i in range(len(self.board)) if not self.checkIfBoardComplete(i)]

    def makeMove(self, move):
        #Make the move
        self.board[move[2]][move[1]][move[0]] = self.player

    def isMoveValid(self, move):
        #Check if the move is valid
        if 0<=move[0]<=2 and 0<=move[1]<=2 and 0<=move[2]<=2:
            if self.board[move[2]][move[1]][move[0]] == "_":
                if move[2] in self.getAvailableBoards():
                    return True
        return False

    def getValidSquarePositions(self):
        positions = []
        for z in range(len(self.board)):
            if z not in self.getAvailableBoards():
                continue
            for y in range(len(self.board[z])):
                for x in range(len(self.board[z][y])):
                    if self.board[z][y][x] == "_":
                        positions.append((x,y,z))
        return positions

    def __repr__(self):
        output = ""
        for z in self.board:
            for y in z:
                output += ",".join(y)+"\n"
            output += "\n"
        return output
        #print(self.board)

class Notactoe:
    def __init__(self):
        self.board = NotactoeBoard()

        self.root = tk.Tk()
        self.imageSizes = (100,100)
        self.images = self.getImages()
        self.grid, self.playerLabel = self.initialiseBoard()

        self.enemy = None

    def getImages(self):
        width, height = self.imageSizes[0], self.imageSizes[1]
        image = Image.open('none.jpg').resize((height, width), Image.ANTIALIAS)
        none = itk.PhotoImage(image)
        image = Image.open('x.jpg').resize((height, width), Image.ANTIALIAS)
        x = itk.PhotoImage(image)
        return (none, x)

    def initialiseBoard(self):
        grid = []
        offset = 0
        for z,board in enumerate(self.board.board):
            gridBoard = []
            offset += 4
            for y,row in enumerate(board):
                gridRow = []
                for x,column in enumerate(row):
                    lbl = tk.Label(self.root,image=self.images[0])
                    lbl.grid(row=y, column=x+offset)
                    lbl.bind('<Button-1>',lambda e,z=z,y=y,x=x: self.onClick(x,y,z,e))
                    gridRow.append(lbl)
                lbl = tk.Label(self.root)
                lbl.grid(row=y,column=offset-1)
                gridBoard.append(gridRow)
            grid.append(gridBoard)
        playerLabel = tk.Label(self.root,text="Player: 1")
        playerLabel.grid(row=3, column=9)

        return grid, playerLabel

    def onClick(self,x,y,z,event): #(x,y,x)
        valid = self.doHumanMove((x,y,z), event, self.images[1])
        #valid = self.doAIMove(self.images[1])
        #valid = self.doRandomMove(self.images[1])
        self.removeCompletedBoards()
        if valid:
            self.changePlayer()

        if valid and self.enemy=="ai":
            #Do the ai move
            self.doAIMove(self.images[1])
            self.removeCompletedBoards()
            self.changePlayer()

        #self.onClick(0,0,0,None)

    def removeCompletedBoards(self):
        for i in range(3):
            if self.board.checkIfBoardComplete(i):
                for y in self.grid[i]:
                    for x in y:
                        x.config(background='Red')

    def doHumanMove(self,move,event,image):
        if self.board.isMoveValid(move):
            self.board.makeMove(move)
            event.widget.config(image=image)
            winner = self.board.checkIfWon()
            #print("Human move: ", winner)
            if winner:
                self.endGame(winner)
            return True
        else:
            return False

    def doAIMove(self,image):
        move = self.getBestMove()
        self.board.makeMove(move)
        self.grid[move[2]][move[1]][move[0]].config(image=image)
        winner = self.board.checkIfWon()
        #print("AI move: ", winner)
        if winner:
            self.endGame(winner)
        return True

    def doRandomMove(self, image):
        move = self.getRandomMove()
        self.board.makeMove(move)
        self.grid[move[2]][move[1]][move[0]].config(image=image)
        winner = self.board.checkIfWon()
        if winner:
            self.endGame(winner)
        return True

    def changePlayer(self):
        self.board.playerNum = (self.board.playerNum+1)%2
        self.playerLabel.config(text="Player: {}".format(self.board.playerNum+1))

    def endGame(self, winner):
        message = "Draw!" if winner=="Draw" else 'Player #{} won!'.format(int(winner)+1)
        messagebox.showinfo('End of play', message)
        self.root.destroy()
        exit()

    def playHuman(self):
        self.enemy = "human"
        self.root.mainloop()

    def playAI(self):
        self.enemy = "ai"
        self.doAIMove(self.images[1])
        self.removeCompletedBoards()
        self.changePlayer()
        #self.onClick(0,0,0,None)
        self.root.mainloop()



    def getRandomMove(self):
        return random.choice(self.board.getValidSquarePositions())

    def getBestMove(self):
        print("Human", self.getBoardPolynomial(self.board.board), targets)

        bestMoves = []
        goodMoves = []
        fairMoves = []
        moves = self.board.getValidSquarePositions()
        for move in moves:
            currentBoard = copy.deepcopy(self.board.board)
            prevAvailableBoards = self.board.getAvailableBoards()
            self.board.makeMove(move)

            boardPolynomial = self.getBoardPolynomial(self.board.board)
            #If it maintains integrity
            if boardPolynomial in targets:
                #print(move, boardPolynomial, self.board)
                goodMoves.append(move)
                #If it can kill a board and maintain integrity
                if self.board.getAvailableBoards() != prevAvailableBoards:
                    bestMoves.append(move)
            #If it doesn't lose the game
            elif self.board.checkIfWon() == False:
                fairMoves.append(move)

            self.board.board = copy.deepcopy(currentBoard)

        if len(bestMoves) != 0:
            move = random.choice(bestMoves)
        elif len(goodMoves) != 0:
            move = random.choice(goodMoves)
        else:
            print("No way to win, if opponent plays optimally")
            if len(fairMoves) != 0:
                move = random.choice(fairMoves)
            else:
                move = random.choice(moves)

        currentBoard = copy.deepcopy(self.board.board)
        self.board.makeMove(move)
        print("AI",self.getBoardPolynomial(self.board.board), targets)
        self.board.board = copy.deepcopy(currentBoard)

        return move

    def getBoardPolynomial(self, board):
        boardPolynomial = ""
        for i in range(len(board)):
            board = self.getBoardTuple(i)
            boardPolynomial += self.getSinglePolynomial(board)
        boardPolynomial = self.reducePolynomial(boardPolynomial)
        return boardPolynomial

    def getValidChildren(self, node):
        children = []
        #Apply one substitution at a time, and defer nested over the tree
        for string,substitution in reductions.items():
            start = node
            for char in node:
                oldString = string
                string = string.replace(char,"",1)
                if oldString != string:
                    start = start.replace(char,"",1)
            if string == "":
                children.append("".join(sorted(start+substitution)))
            else:
                pass 
        return list(set(children))

    def reducePolynomial(self, polynomial):
        polynomial = "".join(sorted(polynomial))
        count, tree, visited = 0, [polynomial], [polynomial]
        while True:
            #print(count, tree)
            next = []
            for node in tree:
                if node in targets:
                    return node
                next.extend(self.getValidChildren(node))
            next = [n for n in next if n not in visited]
            visited.extend(next)
            tree = next[:]
            #If the recursion depth is exceeded
            if count > 3 or len(tree) == 0:
                return sorted(visited, key=lambda x: len(x))[0]
            count += 1

    def getBoardTuple(self, i):
        board = []
        for row in self.board.board[i]:
            newRow = []
            for item in row:
                newRow.append(1 if item == self.board.player else 0)
            board.append(tuple(newRow))
        return tuple(board)

    def getSinglePolynomial(self, board):
        for i in range(4):
            board = tuple(zip(*board[::-1]))
            if board in data:
                return data[board]
        return ""


game = Notactoe()
"""while True:
    print(game.reducePolynomial(input()))
while True:
    print(game.getValidChildren(input()))"""
game.playAI()
