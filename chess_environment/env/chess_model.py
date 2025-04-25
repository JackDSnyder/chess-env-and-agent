#--- Setup ---#
PIECE_MAPPING = {
    0: "..",       # Empty square
    1: "BR",     # Black Rook
    2: "BN",     # Black Knight
    3: "BB",     # Black Bishop
    4: "BQ",     # Black Queen
    5: "BK",     # Black King
    6: "BP",     # Black Pawn
    7: "WR",     # White Rook
    8: "WN",     # White Knight
    9: "WB",     # White Bishop
    10: "WQ",    # White Queen
    11: "WK",    # White King
    12: "WP"     # White Pawn
}
REVERSED_MAPPING = {value: key for key, value in PIECE_MAPPING.items()}
PLAYER_PIECES = {"black": [1, 2, 3, 4, 5, 6], "white": [7, 8, 9, 10, 11, 12], }
UNICODE_MAPPING = {
    7: "\u2656", 8: "\u2658", 9: "\u2657", 10: "\u2655", 11: "\u2654", 12: "\u2659",
    1: "\u265C", 2: "\u265E", 3: "\u265D", 4: "\u265B", 5: "\u265A", 6: "\u265F",
}

class GameState:
    def __init__(self):
        self.enPassantLocation = []
        self.kingMoved = {"white": 0, "black": 0}
        self.rookMoved = {
            "whiteKingside": 0,
            "whiteQueenside": 0,
            "blackKingside": 0,
            "blackQueenside": 0,
        }

def getNewBoard():
    return [
        [1, 2, 3, 4, 5, 3, 2, 1],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [12, 12, 12, 12, 12, 12, 12, 12],
        [7, 8, 9, 10, 11, 9, 8, 7]]

def printBoard(board):
    for row in range(len(board)):
        for tile in board[row]:
            if tile:
                print(UNICODE_MAPPING[tile], end=" ")
            else:
                print(".", end=" ")
        print()

#--- Observe Helpers ---#
def validLocation(location):
    return (0 <= location[0] <= 7) and (0 <= location[1] <= 7)

def getPieceMoves(observation, originalRow, originalCol):
    board = observation["board"]
    agent = "black" if board[originalRow][originalCol] in PLAYER_PIECES["black"] else "white"
    agentPieces = PLAYER_PIECES[agent]
    
    def getRookMoves(board, originalRow, originalCol, agentPieces):
        moves = []
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # Up, down, left, right

        for direction in directions:
            row, col = originalRow, originalCol
            while True:
                row += direction[0]
                col += direction[1]
                location = [row, col]
                if not validLocation(location):
                    break
                tile = board[row][col]
                if tile:
                    if tile not in agentPieces:
                        moves.append(location)  # Capture
                    break
                moves.append(location)
        return moves
    def getBishopMoves(board, originalRow, originalCol, agentPieces):
        moves = []
        directions = [[-1, -1], [-1, 1], [1, -1], [1, 1]]  # Diagonal directions

        for direction in directions:
            row, col = originalRow, originalCol
            while True:
                row += direction[0]
                col += direction[1]
                location = [row, col]
                if not validLocation(location):
                    break
                tile = board[row][col]
                if tile:
                    if tile not in agentPieces:
                        moves.append(location)  # Capture
                    break
                moves.append(location)  
        return moves
    def getQueenMoves(board, originalRow, originalCol, agentPieces):
        rookMoves = getRookMoves(board, originalRow, originalCol, agentPieces)
        bishopMoves = getBishopMoves(board, originalRow, originalCol, agentPieces)
        return rookMoves + bishopMoves
    def getKnightMoves(board, originalRow, originalCol, agentPieces):
        moves = []
        moveMap = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]

        for move in moveMap:
            row = originalRow + move[0]
            col = originalCol + move[1]
            location = [row, col]
            if validLocation(location):
                tile = board[row][col]
                if tile not in agentPieces:
                    moves.append(location)
        return moves
    def getKingMoves(board, originalRow, originalCol, agentPieces):
        moves = []
        moveMap = [
            [1, -1], [1, 0], [1, 1],
            [0, -1],          [0, 1],
            [-1, -1], [-1, 0], [-1, 1]
        ]
        enemyKingCode = 11 if 5 in agentPieces else 5

        # Find the enemy king's position
        enemyKingPos = None
        for r in range(8):
            for c in range(8):
                if board[r][c] == enemyKingCode:
                    enemyKingPos = (r, c)
                    break
            if enemyKingPos:
                break

        for move in moveMap:
            row = originalRow + move[0]
            col = originalCol + move[1]
            location = [row, col]

            if validLocation(location):
                tile = board[row][col]
                if tile not in agentPieces:
                    # Ensure the target square is not adjacent to the enemy king
                    if enemyKingPos:
                        enemyRow, enemyCol = enemyKingPos
                        # Calculate the distance between the target square and the enemy king
                        distanceRow = abs(row - enemyRow)
                        distanceCol = abs(col - enemyCol)
                        if distanceRow <= 1 and distanceCol <= 1:
                            continue  # Skip this move as it places the king adjacent to the enemy king
                    moves.append(location)
                        
        return moves
    def getPawnMoves(board, originalRow, originalCol, agentPieces):
        moves = []
        moveMap = []
        captureMap = []
        agent = "black" if board[originalRow][originalCol] in PLAYER_PIECES["black"] else "white"
        

        if agent == "black":
            moveMap = [[1, 0]]
            captureMap = [[1, -1], [1, 1]]
            if originalRow == 1:
                moveMap.append([2, 0])
        else:
            moveMap = [[-1, 0]]
            captureMap = [[-1, -1], [-1, 1]]
            if originalRow == 6:
                moveMap.append([-2, 0])

        # Single forward move
        oneStepLocation = [originalRow + moveMap[0][0], originalCol + moveMap[0][1]]
        if validLocation(oneStepLocation) and not board[oneStepLocation[0]][oneStepLocation[1]]:
            moves.append(oneStepLocation)

            # Two-square forward move (only if single step is clear)
            if len(moveMap) > 1:
                twoStepLocation = [originalRow + moveMap[1][0], originalCol + moveMap[1][1]]
                if validLocation(twoStepLocation) and not board[twoStepLocation[0]][twoStepLocation[1]]:
                    moves.append(twoStepLocation)

        # Capture moves
        for capture in captureMap:
            location = [originalRow + capture[0], originalCol + capture[1]]
            if validLocation(location):
                tile = board[location[0]][location[1]]
                if tile and tile not in agentPieces:
                    moves.append(location)

        return moves

    getFuncs = {"R": getRookMoves, "B": getBishopMoves, "Q": getQueenMoves,
                "P": getPawnMoves, "N": getKnightMoves, "K": getKingMoves}
    piece = PIECE_MAPPING[board[originalRow][originalCol]]
    moves = getFuncs[piece[1]](board, originalRow, originalCol, agentPieces)
    return moves

def isMoveSafeForKing(endRow, endCol, board, agent):
    opposingKing = 5 if agent == 'black' else 11
    moveMap = [[1, -1], [1, 0], [1, 1], [0, -1], [0, 1], [-1, -1], [-1, 0], [-1, 1]]
    
    for move in moveMap:
        neighborRow = endRow + move[0]
        neighborCol = endCol + move[1]
        if 0 <= neighborRow < 8 and 0 <= neighborCol < 8:
            if board[neighborRow][neighborCol] == opposingKing:
                return False
    return True

def checkCastling(agent, observation):
    board = observation["board"]
    agentRow = 0 if agent == "black" else 7
    if observation["kingMoved"][agent] or board[agentRow][4] != REVERSED_MAPPING[agent[0].upper() + "K"]:
        return [0, 0]

    checkKing = not observation["rookMoved"][f"{agent}Kingside"]
    checkQueen = not observation["rookMoved"][f"{agent}Queenside"]
    kingCastle, queenCastle = 0, 0


    # Check if kingside castling is allowed
    if checkKing:
        if board[agentRow][7] == REVERSED_MAPPING[agent[0].upper() + "R"]:
            # Tiles between are empty
            if board[agentRow][5] == 0 and board[agentRow][6] == 0:
                # Ensure squares are not under attack (add your check here)
                if not inCheck(agent, board):
                    kingCastle = 1

    # Check if queenside castling is allowed
    if checkQueen:
        if board[agentRow][0] == REVERSED_MAPPING[agent[0].upper() + "R"]:
            # Tiles between are empty
            if board[agentRow][1] == 0 and board[agentRow][2] == 0 and board[agentRow][3] == 0:
                # Ensure squares are not under attack (add your check here)
                if not inCheck(agent, board):
                    queenCastle = 1

    return [kingCastle, queenCastle]

def inCheck(agent, board):
    # Find our king
    for row in range(8):
            for col in range(8):
                tile = board[row][col]
                if tile == REVERSED_MAPPING[agent[0].upper()+"K"]:
                    kingRow, kingCol = row, col

    # See if king is targeted
    enemy = "white" if agent == "black" else "black"
    for startRow in range(8):
            for startCol in range(8):
                tile = board[startRow][startCol]
                if tile in PLAYER_PIECES[enemy]:
                    for endPosition in getPieceMoves({"board": board}, startRow, startCol):
                        try:
                            if endPosition == [kingRow, kingCol]:
                                return True
                        except UnboundLocalError:
                            print("No King found")
                            printBoard(board)
                            raise UnboundLocalError

    return False

def maskIndex(startRow, startCol, endRow, endCol, special=None):
    # Promotion moves start at 4096
    if special in [0, 1, 2, 3]:  # 0=queen, 1=rook, 2=bishop, 3=knight
        return 4096 + endCol * 4 + special
    
    # Castling moves start at 4128
    elif special == -1:
        if (startRow, startCol, endRow, endCol) == (7, 4, 7, 6):  # White kingside castling
            return 4128
        elif (startRow, startCol, endRow, endCol) == (7, 4, 7, 2):  # White queenside castling
            return 4129
        elif (startRow, startCol, endRow, endCol) == (0, 4, 0, 6):  # Black kingside castling
            return 4128
        elif (startRow, startCol, endRow, endCol) == (0, 4, 0, 2):  # Black queenside castling
            return 4129
    elif special is None:
        
        # Regular moves: startLocation * 64 + endLocation
        startLocation = startRow * 8 + startCol
        endLocation = endRow * 8 + endCol
        return startLocation * 64 + endLocation

    raise ValueError("Invalid move parameters for action encoding.")

def getActionMask(agent, observation):
    action_mask = [0 for i in range(4130)]
    board = observation["board"]
    enemyKing = 5 if agent == "white" else 11
    
    # Regular and Pawn Moves
    for startRow in range(8):
        for startCol in range(8):
            tile = board[startRow][startCol]
            if tile in PLAYER_PIECES[agent]:
                for endPosition in getPieceMoves(observation, startRow, startCol):
                    endRow, endCol = endPosition[0], endPosition[1]

                    # Can't capture king
                    if board[endRow][endCol] == enemyKing:
                        continue

                    # Check if moving to a square adjacent to the opposing king
                    if tile == enemyKing:  # King piece code for each agent
                        if not isMoveSafeForKing(endRow, endCol, board, agent):
                            continue

                    # Simulate the move
                    simBoard = [row[:] for row in board]
                    simBoard[endRow][endCol] = board[startRow][startCol]
                    simBoard[startRow][startCol] = 0
                    
                    # Only add the move if it does not result in check
                    if not inCheck(agent, simBoard):
                        # Non-pawn moves
                        if tile != 6 and tile != 12:
                            action_mask[maskIndex(startRow, startCol, endRow, endCol)] = 1
                        # Pawn moves, with promotion handling
                        elif tile == 6 or tile == 12:
                            if (agent == "white" and endRow == 0) or (agent == "black" and endRow == 7):
                                for i in range(4):  # 0=queen, 1=rook, 2=bishop, 3=knight
                                    action_mask[maskIndex(startRow, startCol, endRow, endCol, special=i)] = 1
                            else:
                                action_mask[maskIndex(startRow, startCol, endRow, endCol)] = 1

    # Check Castling
    castling = checkCastling(agent, observation)
    kingRow = 0 if agent == "black" else 7
    for i in range(2):
        if castling[i]:
            simBoard = [row[:] for row in board]
            newKingCol = 6 if i == 0 else 2
            simBoard[kingRow][newKingCol] = REVERSED_MAPPING[agent[0].upper() + "K"]
            newRookCol = 5 if i == 0 else 3
            simBoard[kingRow][newRookCol] = REVERSED_MAPPING[agent[0].upper() + "R"]
            simBoard[kingRow][4] = 0
            simBoard[kingRow][7 if i == 0 else 0] = 0
            
            if not inCheck(agent, simBoard):
                action_mask[maskIndex(kingRow, 4, kingRow, newKingCol, special=-1)] = 1

    # En Passant
    if observation["enPassantLocation"]:
        directions = [[0, -1], [0, 1]]
        epRow, epCol = observation["enPassantLocation"][0], observation["enPassantLocation"][1]
        pawnDirection = -1 if agent == "white" else 1
        
        for direction in directions:
            pawnRow, pawnCol = epRow + pawnDirection, epCol + direction[1]
            # Check if pawnRow and pawnCol are within bounds
            if 0 <= pawnRow < 8 and 0 <= pawnCol < 8:
                if board[pawnRow-pawnDirection][pawnCol] == REVERSED_MAPPING[agent[0].upper() + "P"]:
                    simBoard = [row[:] for row in board]
                    simBoard[pawnRow][pawnCol] = 0
                    simBoard[epRow][epCol] = 0
                    simBoard[epRow + pawnDirection][epCol] = REVERSED_MAPPING[agent[0].upper() + "P"]
                    if not inCheck(agent, simBoard):
                        action_mask[maskIndex(pawnRow, pawnCol, epRow + pawnDirection, epCol)] = 1
                    
    return action_mask

#--- Step Helpers ---#
def actionToPositions(action, agent):
    # Promotion moves now start at 4096
    if 4096 <= action < 4128:
        endCol = (action - 4096) // 4
        promotionPiece = (action - 4096) % 4
        startRow = 1 if agent == "white" else 6
        startCol = endCol
        endRow = 7 if startRow == 6 else 0
        return [startRow, startCol], [endRow, endCol], promotionPiece

    # Castling moves are now at 4128 and 4129
    elif action == 4128:
        kingRow = 7 if agent == "white" else 0
        return [kingRow, 4], [kingRow, 6], -1  # Kingside castling
    elif action == 4129:
        kingRow = 7 if agent == "white" else 0
        return [kingRow, 4], [kingRow, 2], -2  # Queenside castling

    # Regular moves
    startSquare = action // 64
    endSquare = action % 64
    startRow, startCol = divmod(startSquare, 8)
    endRow, endCol = divmod(endSquare, 8)
    return [startRow, startCol], [endRow, endCol], None

def remakeGameState(observation):
    newGS = GameState()
    newGS.kingMoved = observation["kingMoved"].copy()
    newGS.rookMoved = observation["rookMoved"].copy()
    return newGS

def makeMove(observation, agent, pieceStart, pieceEnd, specialMove) -> tuple[list, GameState, bool]:
    board = observation["board"]
    if specialMove != -1 and specialMove != -2:
        startRow, startCol = pieceStart[0], pieceStart[1]
        endRow, endCol = pieceEnd[0], pieceEnd[1]
    kingRow = 0 if agent == "black" else 7

    # Castling
    if specialMove == -1:  # Kingside castling
        board[kingRow][6] = REVERSED_MAPPING[agent[0].upper() + "K"]
        board[kingRow][5] = REVERSED_MAPPING[agent[0].upper() + "R"]
        board[kingRow][7] = 0
        board[kingRow][4] = 0
        observation["kingMoved"].update({agent: 1})
        observation["rookMoved"].update({f"{agent}Queenside": 1})
        observation["rookMoved"].update({f"{agent}Kingside": 1})
        return board, remakeGameState(observation), False

    elif specialMove == -2:  # Queenside castling
        board[kingRow][2] = REVERSED_MAPPING[agent[0].upper() + "K"]
        board[kingRow][3] = REVERSED_MAPPING[agent[0].upper() + "R"]
        board[kingRow][0] = 0
        board[kingRow][4] = 0
        observation["kingMoved"].update({agent: 1})
        observation["rookMoved"].update({f"{agent}Queenside": 1})
        observation["rookMoved"].update({f"{agent}Kingside": 1})
        return board, remakeGameState(observation), False

    isPawn = board[startRow][startCol] == REVERSED_MAPPING[agent[0].upper() + "P"]
    
    # Pawn Promotion
    if specialMove is not None and 0 <= specialMove <= 3:
        if startCol-1 >= 0:
            isPawn = isPawn or board[startRow][startCol-1] == REVERSED_MAPPING[agent[0].upper() + "P"]
        if startCol+1 <= 7:
            isPawn = isPawn or board[startRow][startCol+1] == REVERSED_MAPPING[agent[0].upper() + "P"]
        
        if isPawn:
            specialToPiece = {0: "Q", 1: "R", 2: "B", 3: "N"}
            if board[startRow][startCol] == REVERSED_MAPPING[agent[0].upper() + "P"]:
                board[startRow][startCol] = 0
            elif board[startRow][startCol-1] == REVERSED_MAPPING[agent[0].upper() + "P"]:
                board[startRow][startCol-1] = 0
            elif board[startRow][startCol+1] == REVERSED_MAPPING[agent[0].upper() + "P"]:
                board[startRow][startCol+1] = 0
            # print(f"end row: {endRow}, end col: {endCol}")
            # print(f"New piece: {REVERSED_MAPPING[agent[0].upper() + specialToPiece[specialMove]]}")
            board[endRow][endCol] = REVERSED_MAPPING[agent[0].upper() + specialToPiece[specialMove]]
            return board, remakeGameState(observation), True

    resetFifty = False
    # En passant / Update resetFifty
    if isPawn:
        resetFifty = True
        if board[endRow][endCol] == 0 and startCol != endCol:
            board[startRow][endCol] = 0

    # Update gamestate if rook or king moved
    if board[0][0] != REVERSED_MAPPING["BR"]:
        observation["rookMoved"].update({f"blackQueenside": 1})
    if board[0][7] != REVERSED_MAPPING["BR"]:
        observation["rookMoved"].update({f"blackKingside": 1})
    if board[0][4] != REVERSED_MAPPING["BK"]:
        observation["kingMoved"].update({"black": 1})
    if board[7][0] != REVERSED_MAPPING["WR"]:
        observation["rookMoved"].update({f"whiteQueenside": 1})
    if board[7][7] != REVERSED_MAPPING["WR"]:
        observation["rookMoved"].update({f"whiteKingside": 1})
    if board[7][4] != REVERSED_MAPPING["WK"]:
        observation["kingMoved"].update({"white": 1})


    # Captured enemy piece
    enemy = "white" if agent == "black" else "black"
    enemyKingRow = 7 if agent == "black" else 0
    if board[endRow][endCol] in PLAYER_PIECES[enemy]:
        resetFifty = True
        # Update gamestate if enemy rook or king captured
        if endRow == enemyKingRow:
            if endCol == 0:
                observation["rookMoved"].update({f"{enemy}Queenside": 1})  
            elif endCol == 7:
                observation["rookMoved"].update({f"{enemy}Kingside": 1})
            elif endCol == 4:
                observation["kingMoved"].update({enemy: 1})


    # Update game state if pawn double moves
    if isPawn:
        if abs(startRow - endRow) > 1:
            observation["enPassantLocation"] = [endRow, endCol]

    # Regular move
    board[endRow][endCol] = board[startRow][startCol]
    board[startRow][startCol] = 0
    return board, remakeGameState(observation), resetFifty

def insufficientMaterial(board) -> bool:
    bPieces = {"K": 0, "Q": 0, "B": 0, "N": 0, "R": 0, "P": 0}
    wPieces = {"K": 0, "Q": 0, "B": 0, "N": 0, "R": 0, "P": 0}
    for row in range(8):
        for col in range(8):
            tile = board[row][col]
            piece = PIECE_MAPPING[tile]
            if piece[0] == "B":
                bPieces[piece[1]] += 1
            elif piece[0] == "W":
                wPieces[piece[1]] += 1

    # Check if there are no queens, rooks, or pawns
    # if true, only bishops, knights, and kings are left
    pieces = [bPieces, wPieces]
    if ((pieces[0]["Q"] == 0 and pieces[0]["R"] == 0 and pieces[0]["P"] == 0) and
        (pieces[1]["Q"] == 0 and pieces[1]["R"] == 0 and pieces[1]["P"] == 0)):

        # Case 1: King vs. King
        if ((wPieces["B"] == 0 and wPieces["N"] == 0) and (bPieces["B"] == 0 and bPieces["N"] == 0)):
            return True
        # Case 2: K and B vs. K and B
        if ((wPieces["B"] == 1 and wPieces["N"] == 0) and (bPieces["B"] == 1 and bPieces["N"] == 0)):
            return True
        # Case 3: K and N vs. K and N
        if ((wPieces["B"] == 0 and wPieces["N"] == 1) and (bPieces["B"] == 0 and bPieces["N"] == 1)):
            return True
        # Cases 4-7
        for i in range(2):
            team1 = bPieces if i == 0 else wPieces
            team2 = wPieces if i == 0 else bPieces

            # Case 4: K and B vs. K
            if ((team1["B"] == 1 and team1["N"] == 0) and (team2["B"] == 0 and team2["N"] == 0)):
                return True
            # Case 5: K and N vs. K
            if ((team1["B"] == 0 and team1["N"] == 1) and (team2["B"] == 0 and team2["N"] == 0)):
                return True
            # Case 6: K and 2N vs. K
            if ((team1["B"] == 0 and team1["N"] == 2) and (team2["B"] == 0 and team2["N"] == 0)):
                return True
            # Case 7: K and B vs. K and N
            if ((team1["B"] == 1 and team1["N"] == 0) and (team2["B"] == 0 and team2["N"] == 1)):
                return True 
    return False



#-- Agent Model Helpers --#
def ACTIONS(observation, agent):
    actions = []
    agentMask = getActionMask(agent, observation)
    for i, action in enumerate(agentMask):
        if action:
            actions.append(i)
    return actions

def RESULTS(observation, action, agent):
    kingMoved = observation["kingMoved"]
    rookMoved = observation["rookMoved"]
    enPassantLocation = observation["enPassantLocation"]
    observationCopy = {
        "board": [row[:] for row in observation["board"]],
        "kingMoved": {
            "white": 1 if kingMoved["white"] else 0,
            "black": 1 if kingMoved["black"] else 0,
        },
        "rookMoved": {
            "whiteKingside": 1 if rookMoved["whiteKingside"] else 0,
            "whiteQueenside": 1 if rookMoved["whiteQueenside"] else 0,
            "blackKingside": 1 if rookMoved["blackKingside"] else 0,
            "blackQueenside": 1 if rookMoved["blackQueenside"] else 0,
        },
        "enPassantLocation": [enPassantLocation[0], enPassantLocation[1]] if enPassantLocation else []
    }
    startLocation, endLocation, specialMove = actionToPositions(action, agent)
    newBoard, newGS, _ = makeMove(observationCopy, agent, startLocation, endLocation, specialMove)
    return {"board": newBoard,
            "kingMoved": newGS.kingMoved,
            "rookMoved": newGS.rookMoved,
            "enPassantLocation": newGS.enPassantLocation}

def GAME_OVER(observation, agent):
    enemyAgent = "black" if agent == "white" else "white"
    board = observation["board"]
    return (insufficientMaterial(board) or not
            any(getActionMask(enemyAgent, observation)))

def validMoveCount(actionMask):
    count = 0
    for move in actionMask:
        count += 1 if move else 0
    return count

def isCapture(move, observation, agent):
    board = observation["board"]
    enemyAgent = "white" if agent == "black" else "black"
    pieceStart, pieceEnd, specialMove = actionToPositions(move, agent)
    if specialMove is None:
        if board[pieceEnd[0]][pieceEnd[1]] in PLAYER_PIECES[enemyAgent]:
            return True
    return False

def isCheck(move, observation, agent):
    board = observation["board"]
    enemyAgent = "white" if agent == "black" else "black"
    pieceStart, pieceEnd, specialMove = actionToPositions(move, agent)
    if specialMove is None:
        simBoard = [row[:] for row in board]
        simBoard[pieceEnd[0]][pieceEnd[1]] = board[pieceStart[0]][pieceStart[1]]
        simBoard[pieceStart[0]][pieceStart[1]] = 0

        if inCheck(enemyAgent, simBoard):
            return True
    return False

def copyObservation(observation):
    return {
        "action_mask": observation["action_mask"][:],
        "observation": {
            "board": [row[:] for row in observation["observation"]["board"]],
            "kingMoved": {key: val for key, val in observation["observation"]["kingMoved"].items()},
            "rookMoved": {key: val for key, val in observation["observation"]["rookMoved"].items()},
            "enPassantLocation": observation["observation"]["enPassantLocation"][:]
        }
    }

def threatCount(position, board, enemyAgent):
    threats = 0
    row, col = position
    for searchRow in range(8):
        for searchCol in range(8):
            piece = board[searchRow][searchCol]
            if piece in PLAYER_PIECES[enemyAgent]:
                possibleMoves = getPieceMoves({"board": board}, searchRow, searchCol)
                if [row, col] in possibleMoves:
                    threats += 1
    return threats

def getTargetPiece(move, observation, agent):
    _, target, _ = actionToPositions(move, agent)
    if not target or len(target) < 2:
        return None 
    targetRow, targetCol = target[0], target[1]
    board = observation["board"]
    
    if 0 <= targetRow < 8 and 0 <= targetCol < 8:
        return board[targetRow][targetCol] 
    return None

def getMoveEndPosition(move, agent):
    _, endPos, _ = actionToPositions(move, agent)
    return endPos

def isPieceSafeFromKing(position, board, agent):
    row, col = position
    opposingKing = 5 if agent == 'white' else 11
    moveMap = [[1, -1], [1, 0], [1, 1], [0, -1], [0, 1], [-1, -1], [-1, 0], [-1, 1]]

    for move in moveMap:
        neighborRow = row + move[0]
        neighborCol = col + move[1]
        if 0 <= neighborRow < 8 and 0 <= neighborCol < 8:
            if board[neighborRow][neighborCol] == opposingKing:
                return False  # The piece is adjacent to the opposing king
    return True

def isProtected(position, board, agent):
    row, col = position
    friendlyPieces = PLAYER_PIECES[agent]

    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece in friendlyPieces:
                possibleMoves = getPieceMoves({"board": board}, r, c)
                if [row, col] in possibleMoves:
                    return True
    return False

def getStartPiece(move, observation, agent):
    startPos, _, _ = actionToPositions(move, agent)
    if not startPos:
        return None 

    startRow, startCol = startPos
    return observation["board"][startRow][startCol]

def isStalemate(observation, enemyAgent):
    # Check if the enemy has any valid moves
    actionMask = getActionMask(enemyAgent, observation)
    hasLegalMoves = any(actionMask)

    # If no legal moves and the king is not in check, it's a stalemate for the enemy
    return not hasLegalMoves and not inCheck(enemyAgent, observation["board"])

def findKingPosition(observation, agent):
    board = observation["board"]
    kingPiece = 5 if agent == "black" else 11  # Black king is 5, white king is 11

    for row in range(8):
        for col in range(8):
            if board[row][col] == kingPiece:
                return (row, col)
    return None  # Should not happen if the king is on the board

def getPromotionPiece(move, agent):
    # Extract the starting and ending positions along with the special move (promotion or castling) indicator
    _, _, promotionPiece = actionToPositions(move, agent)
    
    # Return the promotion piece if the move is a promotion, otherwise return None
    return promotionPiece if promotionPiece in {0, 1, 2, 3} else None

def isCheckmate(move, observation, agent):
    # Generate the next state after making the move
    nextObservation = RESULTS(observation, move, agent)
    enemyAgent = 'white' if agent == 'black' else 'black'

    # Check if the enemy has any legal moves
    enemyActions = ACTIONS(nextObservation, enemyAgent)
    if not enemyActions:
        # If the enemy has no legal moves and is in check, it's checkmate
        if inCheck(enemyAgent, nextObservation['board']):
            return True
    return False

def isCastle(move, observation, agent):
    # Determine if a move is a castling move
    (startRow, startCol), (endRow, endCol), promotionPiece = actionToPositions(move, agent)
    board = observation["board"]
    piece = board[startRow][startCol]

    # King's piece code
    kingPiece = 5 if agent == "black" else 11

    # Check if the moving piece is the king
    if piece != kingPiece:
        return False

    # Check if the king moves two squares horizontally from its starting position
    if startRow == endRow and abs(endCol - startCol) == 2:
        # Optional: Further verify if the rook is in the correct position
        # and that the path between the king and rook is clear
        return True

    return False

def getThreatenedPositions(observation, agent):
    board = observation['board']
    threatenedPositions = set()
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece in PLAYER_PIECES[agent]:
                possibleMoves = getPieceMoves(observation, row, col)
                for move in possibleMoves:
                    threatenedPositions.add((move[0], move[1]))
    return list(threatenedPositions)

def isThreatened(position, board, enemyAgent):
    row, col = position
    for searchRow in range(8):
        for searchCol in range(8):
            piece = board[searchRow][searchCol]
            if piece in PLAYER_PIECES[enemyAgent]:
                possibleMoves = getPieceMoves({"board": board}, searchRow, searchCol)
                if [row, col] in possibleMoves:
                    return True
    return False

def printMove(action, agent):
    if action == 4128:
        print(f"{agent} performed a kingside castle.")
        return
    if action == 4129:
        print(f"{agent} performed a queenside castle.")
        return
    startPos, endPos, special = actionToPositions(action, agent)
    specialToPiece = {0:"Queen", 1:"Rook", 2:"Bishop", 3:"Knight"}
    colToFile = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}
    printStatement = f"{agent} moved piece from {colToFile[startPos[1]]}{8-startPos[0]} to {colToFile[endPos[1]]}{8-endPos[0]}."
    if special is not None and 0 <= special <= 3:
        printStatement = printStatement[:-1] + f" and promoted to a {specialToPiece[special]}"
    print(printStatement)

