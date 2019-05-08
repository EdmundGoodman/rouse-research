#http://cs.williams.edu/~freund/cs136-073/GardnerHexapawn.pdf
import random
import copy
import matplotlib.pyplot as plt
random.seed(0)

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

    def __repr__(self):
        return "\n".join(" ".join(y) for y in self._board)

class Hexapawn:
    def __init__(self, players=["X", "O"]):
        self._players = players
        self._board = HexapawnBoard(self._players)
        self.resetGame()

    def resetGame(self):
        self._playerNum = 0
        self._count = 0
        self._board.resetBoard()

    def getHumanMove(self):
        print()
        while True:
            try:
                oldPos = [int(x) for x in input("Position [x,y] of peice: ").split(",")]
                newPos = [int(x) for x in input("Position [x,y] of position: ").split(",")]
            except TypeError:
                print("Enter 2 numbers, seperated by a comma with no spaces")
                continue
            isMoveInvalid = self._board.isMoveInvalid(oldPos, newPos, self._playerNum)
            if isMoveInvalid:
                print(isMoveInvalid)
            else:
                break
        return oldPos, newPos

    def playHuman(self):
        while True:
            print("{}'s turn:".format(self._players[self._playerNum]))
            print(self._board)
            oldPos, newPos = self.getHumanMove()
            self._board.makeMove(oldPos, newPos)
            winner = self._board.checkIfWon(self._playerNum)
            if winner:
                break
            self._count += 1
            self._playerNum = (self._count)%2
        return "{} Won".format(winner)

    def getRandomMove(self):
        move = random.choice(self._board.getValidMoves(self._playerNum))
        return move[0], move[1]

    def trainMatchboxComputer(self, noIterations):
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
                    oldPos, newPos = self.getRandomMove()
                self._board.makeMove(oldPos, newPos)
                winner = self._board.checkIfWon(self._playerNum)
                if winner:
                    break
                self._count += 1
                self._playerNum = (self._count)%2

            won = True if winner==self._players[1] else False
            m.updateWeights(won)

            offset = 1 if won else -1 #0
            winLossList.append(winLossList[-1]+offset)

        plotGraph(winLossList)

        return m

    def playRobot(self):
        m = self.trainMatchboxComputer(3000)
        self.resetGame()
        m.resetGame()
        while True:
            print("{}'s turn:".format(self._players[self._playerNum]))
            print(self._board)
            if self._playerNum == 1:
                oldPos, newPos = m.getComputerMove(self._board,
                                                   self._playerNum,
                                                   self._count)
            else:
                oldPos, newPos = self.getHumanMove()
            self._board.makeMove(oldPos, newPos)
            winner = self._board.checkIfWon(self._playerNum)
            if winner:
                break
            self._count += 1
            self._playerNum = (self._count)%2
        print("\n\n{}".format(self._board))
        return "{} Won".format(winner)

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

    def getNextBoard(self, board, move):
        boardCopy = copy.deepcopy(board.getBoard())
        boardCopy[move[1][1]][move[1][0]] = boardCopy[move[0][1]][move[0][0]]
        boardCopy[move[0][1]][move[0][0]] = "-"
        return boardCopy

    def getComputerMove(self, board, playerNum, count):
        boardTuple = tuple([tuple(elem) for elem in board.getBoard()])
        self._openBoxes.append(boardTuple)

        if boardTuple not in self.boardStates:
            #If we haven't encountered this move before
            moves = board.getValidMoves(playerNum)
            defaultWeight = {1:4, 3:3, 5:2, 7:1}[count]
            movesWeights = {move:defaultWeight for move in moves}
            self.boardStates[boardTuple] = movesWeights
            #moves *= defaultWeight
        else:
            movesWeights = self.boardStates[boardTuple]
            moves = []
            for key,value in movesWeights.items():
                moves.extend([key]*value)

        if len(moves) == 0:
            self._resigned = True
            return "Resign", "Resign"

        move = random.choice(moves)
        self._movesMade.append(move)
        return move

def plotGraph(l):
    plt.plot(l)
    plt.xlabel('No games')
    plt.ylabel('Win/Losses')
    plt.show()


game = Hexapawn()
print( game.playRobot() )
