#http://cs.williams.edu/~freund/cs136-073/GardnerHexapawn.pdf
from os import system
import random

class gameBoard:
    def __init__(self):
        self.board = [["X","X","X"], ["-","-","-"], ["O","O","O"]]
        self.players = ["X", "O"]
        self.currentPlayer = "X"

    def displayBoard(self):
        #Display the board
        for y in self.board:
            for x in y:
                print(x, end=" ")
            print()
        print("\n")

    def updatePlayer(self):
        #Increment the current player
        self.currentPlayer = self.players[(self.players.index(self.currentPlayer)+1)%2]

    def makeMove(self, oldPos, newPos):
        #Move the piece at oldPos to
        self.board[newPos[1]][newPos[0]] = self.board[oldPos[1]][oldPos[0]]
        self.board[oldPos[1]][oldPos[0]] = "-"

    def getMove(self):
        while True:
            #Get & validate the selected piece
            piece = [int(x)-1 for x in input("Piece position [x,y]: ").split(",")]
            if game.board[piece[1]][piece[0]] != game.currentPlayer:
                print("That's not a valid piece! Please try again")
                continue

            #Get & validate the selected move
            move = [int(x)-1 for x in input("Move position [x,y]: ").split(",")]
            #If the move is to off the board
            if not 0<=move[0]<=2 or not 0<=move[1]<=2:
                print("That's not a valid move! Please try again")
            #If the move isn't diagonal and the move position is occupied
            elif move[0]==piece[0] and game.board[move[1]][move[0]] != "-":
                print("That's not a valid move! Please try again")
            #If the move is diagonal and the move position isn't occupied
            elif move[0]!=piece[0] and game.board[move[1]][move[0]] == "-":
                print("That's not a valid move! Please try again")
            #If the move is more than one y step
            elif abs(move[1]-piece[1]) != 1:
                print("That's not a valid move! Please try again")
            else:
                return piece, move

    def hasWon(self):
        #If a player is at the opposite end
        if len([x for x in self.board[2] if x != self.players[0]]) == 1:
            return self.players[0]
        elif len([x for x in self.board[0] if x != self.players[1]]) == 1:
            return self.players[1]

        #If all enemy pieces are captured
        if self.players[1] not in sum(self.board, []):
            return self.players[0]
        elif self.players[1] not in sum(self.board, []):
            return self.players[0]

        #If player cannot move
        #Player X
        move, pieces = False, []
        for b,y in enumerate(self.board):
            for a,x in enumerate(y):
                if x == self.players[0]:
                    pieces.append([a,b])
        for piece in pieces:
            if self.board[piece[1]+1][piece[0]] == "-":
                move = True
            elif piece[0]+1<=2 and self.board[piece[1]+1][piece[0]+1] == self.players[1]:
                move = True
            elif piece[0]-1>=0 and self.board[piece[1]+1][piece[0]-1] == self.players[1]:
                move = True
        if move == False:
            return self.players[1]

        #Player O
        move, pieces = False, []
        for b,y in enumerate(self.board):
            for a,x in enumerate(y):
                if x == self.players[1]:
                    pieces.append([a,b])
        for piece in pieces:
            if self.board[piece[1]-1][piece[0]] == "-":
                move = True
            elif piece[0]+1<=2 and self.board[piece[1]-1][piece[0]+1] == self.players[0]:
                move = True
            elif piece[0]-1>=0 and self.board[piece[1]-1][piece[0]-1] == self.players[0]:
                move = True
        if move == False:
            return self.players[0]

        #Otherwise, the game goes on
        return False

game = gameBoard()
while not game.hasWon():
    #Display the player and the board
    system('clear')
    print("{}'s turn:\n".format(game.currentPlayer))
    game.displayBoard()

    #Get and make the move, then change the player
    piece, move = game.getMove()
    game.makeMove(piece, move)
    game.updatePlayer()

#Display the final board state, and the winner
system('clear')
game.displayBoard()
print("Player {} won!".format(game.hasWon()))
