#Edmund Goodman - Creative Commons Attribution-NonCommercial-ShareAlike 2.5
from random import choice
import os, time

class tictactoe:
    def __init__(self):
        self.board = [["_" for _ in range(3)] for _ in range(3)]
        self.n = 0
        self.players = ["O", "X"]

    def displayBoard(self):
        for y in self.board:
            print(y)

    def makeMove(self, player, move):
        self.board[move[1]][move[0]] = player

    def checkIfWon(self):
        b = self.board
        for p in self.players:
            if (b[2][0] == p and b[2][1] == p and b[2][2] == p): return p
            elif (b[1][0] == p and b[1][1] == p and b[1][2] == p): return p
            elif (b[0][0] == p and b[0][1] == p and b[0][2] == p): return p
            elif (b[2][0] == p and b[1][0] == p and b[0][0] == p): return p
            elif (b[2][1] == p and b[1][1] == p and b[0][1] == p): return p
            elif (b[2][2] == p and b[1][2] == p and b[0][2] == p): return p
            elif (b[2][0] == p and b[1][1] == p and b[0][2] == p): return p
            elif (b[2][2] == p and b[1][1] == p and b[0][0] == p): return p
        return False

    def isMoveValid(self, position):
        if 0<=position[0]<=2 and 0<=position[0]<=2:
            if self.board[position[1]][position[0]] == "_":
                return True
        return False

    def play(self):
        while True:
            while True:
                os.system('clear')
                self.displayBoard()
                move = [int(x) for x in input("Position x,y: ").split(',')]
                if self.isMoveValid(move):
                    break
                else:
                    print("Invalid Move")
            self.makeMove(self.players[self.n], move)
            self.n = (self.n+1)%2
            if self.checkIfWon():
                print("Player {} Won".format(self.checkIfWon()))
                break
            elif set([x for sublist in self.board for x in sublist]).isdisjoint(set("_")):
                print("Draw")
                break

game = tictactoe()
game.play()
