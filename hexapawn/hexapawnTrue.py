#http://cs.williams.edu/~freund/cs136-073/GardnerHexapawn.pdf
import random, copy
import matplotlib.pyplot as plt

random.seed(0)

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk as itk


class HexapawnBoard:
    def __init__(self, players=["X", "O"]):
        self._players = players
        self.resetBoard()

    def resetBoard(self):
        self._board = [[self._players[0]]*3, ["-","-","-"], [self._players[1]]*3]

    def getPlayers(self):
        return self._players

    def getBoard(self):
        return self._board

    def makeMove(self, oldPos, newPos):
        self._board[newPos[1]][newPos[0]] = self._board[oldPos[1]][oldPos[0]]
        self._board[oldPos[1]][oldPos[0]] = "-"

    def isMoveInvalid(self, oldPos, newPos, playerIndex):
        player = self._players[playerIndex]
        enemy = self._players[(playerIndex+1)%2]

        if self._board[oldPos[1]][oldPos[0]] != player:
            return "You must move your own player"

        offset = 1 if playerIndex==0 else -1
        if oldPos[1]+offset != newPos[1]:
            return "You must move forward 1 step"

        if oldPos[0] != newPos[0]:
            if abs(newPos[0]-oldPos[0]) != 1:
                return "You can only take one square diagonally"
            elif self._board[newPos[1]][newPos[0]] != enemy:
                return "You cannot move diagonally unless you are taking a peice"
        else:
            if self._board[newPos[1]][newPos[0]] == enemy:
                return "You cannot take a peice in front of you"

        return False

    def checkIfWon(self, playerIndex):
        enemyIndex = (playerIndex+1)%2
        player = self._players[playerIndex]
        enemy = self._players[enemyIndex]

        #Advancing a pawn to the third
        y = 0 if playerIndex else 2
        for x in range(3):
            if self._board[y][x] == player:
                return player

        #Capturing all enemy peices
        enemyPositions = self.getPlayerPeices(enemyIndex)
        if len(enemyPositions) == 0:
            return player

        #Acheiving a position where the enemy can't move
        validEnemyMoves = self.getValidMoves(enemyIndex)
        if len(validEnemyMoves) == 0:
            return player

        return False

    def getPlayerPeices(self, n):
        player = self._players[n]
        positions = []
        for y,row in enumerate(self._board):
            for x,item in enumerate(row):
                if item == player:
                    positions.append((x,y))
        return positions

    def getValidMoves(self, n):
        validMoves = []
        for pos in self.getPlayerPeices(n):
            for y in range(3):
                for x in range(3):
                    move = self.isMoveInvalid(pos,(x,y),n)
                    if not move:
                        validMoves.append((pos, (x,y)))
        return validMoves

    def ownsPeice(self, n, position):
        return self._board[position[1]][position[0]] == self._players[n]


    def __repr__(self):
        return "\n".join(" ".join(y) for y in self._board)

class Hexapawn:
    def __init__(self, players=["X", "O"]):
        self._players = players
        self._board = HexapawnBoard(self._players)
        self.resetGame()

        self.selected = []
        self.enemy = None

    def createWindow(self):
        self.root = tk.Tk()
        self.imageSizes = (200,200)
        self.images = self.getImages()
        self.grid = self.initialiseBoard()

    def resetGame(self):
        self._playerNum = 0
        self._count = 0
        self._board.resetBoard()

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
        for i,row in enumerate(self._board.getBoard()):
            gridRow = []
            for j,column in enumerate(row):
                if column == "-":
                    lbl = tk.Label(self.root,image=self.images[0])
                elif column == "X":
                    lbl = tk.Label(self.root,image=self.images[2])
                else:
                    lbl = tk.Label(self.root,image=self.images[1])
                lbl.grid(row=i, column=j, padx=1, pady=1)
                lbl.bind('<Button-1>',lambda e,i=i,j=j: self.onClick(i,j,e))
                gridRow.append(lbl)
            grid.append(gridRow)
        return grid

    def onClick(self,j,i,event):
        #print(self._board)
        #print(self.enemy, self._playerNum)

        #Select the piece to move
        if len(self.selected) == 0:
            #Check if move is valid
            if self._board.ownsPeice(self._playerNum, (i,j)):
                #Highlight the peice on the board, and store the its position
                event.widget.config(background='Red')
                self.selected.append((i,j))

        #Make the move
        elif len(self.selected) == 1:
            #Store the position to move to
            self.selected.append((i,j))

            #Make the human move
            failed = self.makeHumanMove()

            if not failed:
                if self.enemy == "random":
                    self.makeRandomMove()
                elif self.enemy == "human":
                    pass
                elif str(self.enemy) == "ai":
                    self.makeAImove()

    def makeHumanMove(self):
        #Check if the move is valid
        isMoveInvalid = self._board.isMoveInvalid(
            self.selected[0], self.selected[1], self._playerNum)

        #If the move is valid
        if not isMoveInvalid:
            self.makeMove(self.selected)

        #Stop highlighting the peice, and stop storing its position
        self.grid[self.selected[0][1]][self.selected[0][0]].config(
            background='White')
        self.selected = []

        return isMoveInvalid

    def makeRandomMove(self):
        move = random.choice(self._board.getValidMoves(self._playerNum))
        self.makeMove(move)
        return 0

    def makeAImove(self):
        move = self.enemy.getComputerMove(self._board, self._playerNum, self._count)
        self.makeMove(move)
        return 0

    def makeMove(self, move):
        self._board.makeMove(move[0], move[1])

        #print(self._board, "\n", move)
        #Update the board to show the new peice position
        playerImage = self.images[2] if self._playerNum==0 else self.images[1]
        self.grid[move[0][1]][move[0][0]].config(image=self.images[0])
        self.grid[move[1][1]][move[1][0]].config(image=playerImage)

        #If the move won, end the game
        winner = self._board.checkIfWon(self._playerNum)
        if winner:
            self.endGame(winner)

        #Increment game variables
        self._count += 1
        self._playerNum = (self._count)%2
        return 0

    def endGame(self, winner):
        #print(self.board)
        message = "Draw!" if winner=="Draw" else 'Player {} won!'.format(winner)
        messagebox.showinfo('End of play', message)
        self.root.destroy()
        exit()

    def playHuman(self):
        self.createWindow()
        self.enemy = "human"
        self.root.mainloop()

    def playRandom(self):
        self.createWindow()
        self.enemy = "random"
        self.root.mainloop()


    def trainMatchboxComputer(self, noIterations=35000):
        lostCount = 0

        m = MatchboxComputer()
        winLossList = [0]
        for iteration in range(noIterations):
            self.resetGame()
            while True:
                if self._playerNum == 1:
                    oldPos, newPos = m.getComputerMove(self._board,self._playerNum,self._count)
                    if oldPos == newPos == "Resign":
                        winner = self._players[(self._playerNum+1)%2]
                        break
                else:
                    oldPos, newPos = random.choice(self._board.getValidMoves(self._playerNum))
                self._board.makeMove(oldPos, newPos)
                winner = self._board.checkIfWon(self._playerNum)
                if winner:
                    break
                self._count += 1
                self._playerNum = (self._count)%2

            won = True if winner==self._players[1] else False
            m.updateWeights(won)

            if not won: lostCount += 1

            offset = 1 if won else -20 #1
            winLossList.append(winLossList[-1]+offset)

        MatchboxComputer.plotGraph(winLossList)

        print(lostCount)
        print(len(m.boardStates))
        return m


    def playAI(self):
        self.enemy = self.trainMatchboxComputer()
        #return None
        self.resetGame()
        self.enemy.resetGame()
        self.createWindow()
        self.root.mainloop()

class MatchboxComputer:
    def __init__(self):
        self.boardStates = {}
        self.resetGame()

    def resetGame(self):
        self._openBoxes = []
        self._movesMade = []
        self._resigned = False

    def updateWeights(self, won):
        if not won:
            index = -2 if self._resigned else -1
            self.boardStates[self._openBoxes[index]][self._movesMade[-1]] += -1
        self.resetGame()
        self.cleanUpBoardStates()

    def cleanUpBoardStates(self):
        newBoardStates = {}
        for Mkey in self.boardStates.keys():
            newMovesWeights = {}
            for key, value in self.boardStates[Mkey].items():
                if value > 0:
                    newMovesWeights[key] = value
            newBoardStates[Mkey] = newMovesWeights
        self.boardStates = newBoardStates

    def getMirroredBoard(self, board):
        newBoard = []
        for row in board:
            newBoard.append(tuple(reversed(row)))
        return tuple(newBoard)

    def getMirroredMove(self, move):
        return ((2-move[0][0], move[0][1]),(2-move[1][0], move[1][1]))

    def getComputerMove(self, board, playerNum, count):
        boardTuple = tuple([tuple(elem) for elem in board.getBoard()])
        mirrorTuple = self.getMirroredBoard(boardTuple)

        mirrored = mirrorTuple in self.boardStates

        if (boardTuple not in self.boardStates) and (mirrorTuple not in self.boardStates):
            #If we haven't encountered this move before
            self._openBoxes.append(boardTuple)
            moves = board.getValidMoves(playerNum)
            defaultWeight = {1:4, 3:3, 5:2, 7:1}[count]
            movesWeights = {move:defaultWeight for move in moves}
            self.boardStates[boardTuple] = movesWeights
        else:
            #If it is a normal board
            if mirrored:
                boardTuple = mirrorTuple

            self._openBoxes.append(boardTuple)
            movesWeights = self.boardStates[boardTuple]
            moves = []
            for key,value in movesWeights.items():
                moves.extend([key]*value)

        if len(moves) == 0:
            self._resigned = True
            return "Resign", "Resign"

        move = random.choice(moves)
        self._movesMade.append(move)
        if mirrored:
            return self.getMirroredMove(move)
        else:
            return move

    @staticmethod
    def plotGraph(l):
        plt.plot(l)
        plt.xlabel('# Games')
        plt.ylabel('# Wins - # Losses')
        plt.show()

    def __repr__(self):
        return "ai"




game = Hexapawn()
game.playAI()
"""
#Test many random seeds, to check if the number of losses/boxes is consistent
for i in range(30):
    print("Seed: ", i)
    random.seed(i)
    game.playAI()"""
