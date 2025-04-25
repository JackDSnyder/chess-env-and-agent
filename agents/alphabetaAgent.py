import chess_environment.env.chess_model as chess_model
import time

class Agent:
    def __init__(self):
        self.transpositionTable = {}

    def reset(self):
        self.transpositionTable = {}

    # Return best action found after searching for a specific time or minimum depth
    def agent_function(self, observation, agent, turnTime=12):
        startTime = time.time()
        bestActionSoFar = None
        maxSearchDepth = 49
        self.minSearchDepth = 4

        # Iterative deepening search
        for searchDepth in range(self.minSearchDepth, maxSearchDepth):
            # Note: Transposition table is persisted across depths for efficiency
            obsCopy = chess_model.copyObservation(observation)  # Copy observation to avoid modifying the original

            # Perform alpha-beta search at the current depth
            currentAction = self.alphaBetaSearch(obsCopy["observation"], agent, searchDepth, startTime, turnTime)

            # Update the best action found if search completes in time
            if currentAction is not None:
                bestActionSoFar = currentAction

            # Check if time limit is reached
            if time.time() - startTime > turnTime:
                chess_model.printMove(bestActionSoFar, agent)
                return bestActionSoFar

        chess_model.printMove(bestActionSoFar, agent)
        return bestActionSoFar

    def alphaBetaSearch(self, observation, agent, maxDepth, startTime, turnTime):
        alpha, beta = -float("inf"), float("inf")
        bestValue, bestAction = -float("inf"), None
        moves = self.getOrderedMoves(observation, agent)

        if not moves:
            return None

        for i, action in enumerate(moves):
            # Time check for deep searches
            if maxDepth > self.minSearchDepth and time.time() - startTime > turnTime:
                return None  # Exit if time limit is exceeded
            # Generate the next state
            nextObservation = chess_model.RESULTS(observation, action, agent)
            enemyAgent = "white" if agent == "black" else "black"

            # Decide whether to store the state in the transposition table
            fullEval = i < 3 # Fully evaluate top moves
            stateHash = None
            if fullEval or maxDepth > 2:
                stateHash = self.hashState(observation, agent)

            # Check if state is already evaluated
            if stateHash and stateHash in self.transpositionTable:
                value = self.transpositionTable[stateHash]
            else:
                # Recursively evaluate the move using minValue
                value = self.minValue(nextObservation, enemyAgent, 1, alpha, beta, maxDepth, startTime, turnTime)
                # Store the evaluated value in the transposition table
                if fullEval and stateHash is not None:
                    self.transpositionTable[stateHash] = value

            # Check for time cutoff
            if value is None:
                return None

            # Update best value and action if a better move is found
            if value > bestValue:
                bestValue, bestAction = value, action
                alpha = max(alpha, bestValue)  # Update alpha

            # Alpha-beta pruning cutoff
            if alpha >= beta:
                break

        return bestAction

    def maxValue(self, observation, agent, depth, alpha, beta, maxDepth, startTime, turnTime):
        # Time check for deeper levels
        if depth > self.minSearchDepth and time.time() - startTime > turnTime:
            return None  # Time limit exceeded

        stateHash = self.hashState(observation, agent)
        if stateHash in self.transpositionTable:
            return self.transpositionTable[stateHash]  # Return cached value

        # Use evaluation at maximum depth or if game is over
        if depth >= maxDepth or chess_model.GAME_OVER(observation, agent):
            evalValue = self.evaluate(observation, agent)
            self.transpositionTable[stateHash] = evalValue
            return evalValue

        bestValue = -float("inf")
        # Iterate over all possible moves
        for action in self.getOrderedMoves(observation, agent):
            nextObservation = chess_model.RESULTS(observation, action, agent)
            enemyAgent = "white" if agent == "black" else "black"
            # Recursively call minValue for the opponent's response
            value = self.minValue(nextObservation, enemyAgent, depth + 1, alpha, beta, maxDepth, startTime, turnTime)

            # Check for time cutoff
            if value is None:
                return None

            bestValue = max(bestValue, value)
            alpha = max(alpha, bestValue)  # Update alpha

            # Alpha-beta pruning cutoff
            if bestValue >= beta:
                break

        # Cache the evaluated value
        self.transpositionTable[stateHash] = bestValue
        return bestValue

    def minValue(self, observation, agent, depth, alpha, beta, maxDepth, startTime, turnTime):
        # Time check for deeper levels
        if depth > self.minSearchDepth and time.time() - startTime > turnTime:
            return None  # Time limit exceeded

        stateHash = self.hashState(observation, agent)
        if stateHash in self.transpositionTable:
            return self.transpositionTable[stateHash]  # Return cached value

        # Use evaluation at maximum depth or if game is over
        if depth >= maxDepth or chess_model.GAME_OVER(observation, agent):
            evalValue = self.evaluate(observation, agent)
            self.transpositionTable[stateHash] = evalValue
            return evalValue

        bestValue = float("inf")
        # Iterate over all possible moves
        for action in self.getOrderedMoves(observation, agent):
            nextObservation = chess_model.RESULTS(observation, action, agent)
            enemyAgent = "white" if agent == "black" else "black"
            # Recursively call maxValue for the agent's response
            value = self.maxValue(nextObservation, enemyAgent, depth + 1, alpha, beta, maxDepth, startTime, turnTime)

            # Check for time cutoff
            if value is None:
                return None

            bestValue = min(bestValue, value)
            beta = min(beta, bestValue)  # Update beta

            # Alpha-beta pruning cutoff
            if bestValue <= alpha:
                break

        # Cache the evaluated value
        self.transpositionTable[stateHash] = bestValue
        return bestValue

    def getOrderedMoves(self, observation, agent):
        # Get all legal actions for the agent
        moves = chess_model.ACTIONS(observation, agent)
        prioritizedMoves = []

        # Assign a priority score to each move
        for move in moves:
            priority = self.movePriority(move, observation, agent)
            prioritizedMoves.append((move, priority))
            # startPos, endPos, _ = chess_model.actionToPositions(move, agent)

        # Sort moves by priority in descending order
        prioritizedMoves.sort(key=lambda x: x[1], reverse=True)

        # Return the list of moves in order of priority
        return [move for move, _ in prioritizedMoves]

    def movePriority(self, move, observation, agent):
        board = observation["board"]
        enemyAgent = "white" if agent == "black" else "black"
        startPiece = chess_model.getStartPiece(move, observation, agent)
        targetPiece = chess_model.getTargetPiece(move, observation, agent)
        (startRow, startCol), (endRow, endCol), promotionPiece = chess_model.actionToPositions(move, agent)

        agentMaterial = self.materialEvaluation(observation, agent)
        enemyMaterial = self.materialEvaluation(observation, enemyAgent)
        totalMaterial = agentMaterial + enemyMaterial

        # Determine game phase
        if totalMaterial <= 24 or agentMaterial <= 15 or enemyMaterial <= 15:
            gamePhase = "endgame"
        elif totalMaterial >= 60:
            gamePhase = "opening"
        else:
            gamePhase = "midgame"

        priority = 0

        # Universal priorities
        # 1. Checkmate
        if chess_model.isCheckmate(move, observation, agent):
            return 100000 

        # 2. Promotion
        if promotionPiece is not None:
            if promotionPiece in [4, 10]:  # Queen pieces
                priority += 900 
            else:
                priority += 800  

        # 3. Captures
        if chess_model.isCapture(move, observation, agent):
            captureValue = self.getPieceValue(targetPiece, enemyAgent)
            attackerValue = self.getPieceValue(startPiece, agent)
            materialGain = captureValue - attackerValue
            priority += 500 + (materialGain * 10)
            # Bonus if the capture is safe
            nextObservation = chess_model.RESULTS(observation, move, agent)
            if not chess_model.isThreatened((endRow, endCol), nextObservation["board"], enemyAgent):
                priority += 50

        # 4. Checks (Conditional)
        if chess_model.isCheck(move, observation, agent):
            if self.isAdvantageousCheck(move, observation, agent):
                priority += 300  # Higher bonus for advantageous checks
            else:
                priority += 50  # Lower bonus for non-advantageous checks

        # Game-phase-specific priorities
        if gamePhase == "opening":
            # Develop Knights and Bishops
            if startPiece in [2, 3, 8, 9]:  # Knights and Bishops
                if (startRow == (0 if agent == "black" else 7)):
                    priority += 200
                else:
                    priority += 50  # Less priority if already developed
            # Central pawn advances
            if startPiece in [6, 12]:  # Pawns
                if startCol in [3, 4]:
                    priority += 100  # Encourage central pawns
            # Early castling
            if chess_model.isCastle(move, observation, agent):
                nextObs = chess_model.RESULTS(observation, move, agent)
                queenPos = self.findQueenPosition(nextObs, agent)
                if queenPos and chess_model.isThreatened(queenPos, nextObs["board"], enemyAgent):
                    priority -= 100
                else:
                    priority += 250
            # Discourage early queen moves
            if startPiece in [4, 10]:  # Queen
                priority -= 100

        elif gamePhase == "midgame":
            # Control open files with rooks
            if startPiece in [1, 7]:  # Rooks
                if self.isOpenFile(endCol, board):
                    priority += 150
            # Create threats to higher-value pieces
            if self.isThreateningHigherValuePiece(move, observation, agent):
                priority += 100
            # Encourage piece activity
            if self.isInEnemyTerritory(endRow, agent):
                priority += 50

        elif gamePhase == "endgame":
            # Activate the king
            if startPiece in [5, 11]:  # King
                priority += 200
            # Push passed pawns
            if self.isPassedPawn((endRow, endCol), board, agent):
                priority += 300
            # Bring pieces closer to the enemy king
            if self.isCloseToEnemyKing((endRow, endCol), observation, agent):
                priority += 100

        # Default priority for other moves
        priority += 10

        # Penalize moves that leave the piece under attack
        nextObservation = chess_model.RESULTS(observation, move, agent)
        if chess_model.isThreatened((endRow, endCol), nextObservation["board"], enemyAgent):
            priority -= 500 

        # Additional Safeguard: Avoid moves that leave the king in check
        if not self.isMoveSafe(move, observation, agent):
            priority -= 1000 

        return priority

    def hashState(self, observation, agent):
        return hash((tuple(map(tuple, observation["board"])), agent))

    def evaluate(self, observation, agent):
        board = observation["board"]
        enemyAgent = "white" if agent == "black" else "black"

        pieceValues = {
            0: 0,    # Empty square
            1: 6,    # Black Rook
            2: 3,    # Black Knight
            3: 3,    # Black Bishop
            4: 9,    # Black Queen
            5: 0,    # Black King
            6: 1,    # Black Pawn
            7: 6,    # White Rook 
            8: 3,    # White Knight
            9: 3,    # White Bishop
            10: 9,   # White Queen
            11: 0,   # White King
            12: 1    # White Pawn
        }
        agentPieces = set(chess_model.PLAYER_PIECES[agent])
        enemyPieces = set(chess_model.PLAYER_PIECES[enemyAgent])

        # Initialize evaluation components
        agentMaterial = 0
        enemyMaterial = 0
        mobility = 0
        safetyScore = 0
        pawnAdvancementReward = 0
        centerControlReward = 0
        rookAndBishopDevelopment = 0
        knightDevelopmentReward = 0
        kingPositioningReward = 0
        boardCoverageReward = 0
        livingQueenBonus = 0
        activationBonus = 0
        captureRiskEvaluation = 0
        winLoss = 0
        stalematePenalty = 0

        # Center reward helpers
        centerValues = [
            [1, 1, 2, 3, 3, 2, 1, 1],
            [1, 2, 3, 4, 4, 3, 2, 1],
            [2, 3, 4, 5, 5, 4, 3, 2],
            [3, 4, 5, 6, 6, 5, 4, 3],
            [3, 4, 5, 6, 6, 5, 4, 3],
            [2, 3, 4, 5, 5, 4, 3, 2],
            [1, 2, 3, 4, 4, 3, 2, 1],
            [1, 1, 2, 3, 3, 2, 1, 1]]
        pieceMultipliers = {
            1: 5,    # Black Rook
            2: 3,    # Black Knight
            3: 3,    # Black Bishop
            4: 9,    # Black Queen
            6: 1,    # Black Pawn
            7: 5,    # White Rook
            8: 3,    # White Knight
            9: 3,    # White Bishop
            10: 9,   # White Queen
            12: 1    # White Pawn
        }

        allActions = chess_model.ACTIONS(observation, agent)
        mobility = chess_model.validMoveCount(chess_model.getActionMask(agent, observation)) * 0.5  # mobilityWeight

        # Initialize list to store pawn rows for dynamic reward calculation
        pawnRows = []

        # Traverse the board once
        for row in range(8):
            for col in range(8):
                piece = board[row][col]

                if piece in agentPieces:
                    pieceValue = pieceValues[piece]
                    agentMaterial += pieceValue

                    # Safety Evaluation
                    isThreatened = chess_model.isThreatened((row, col), board, enemyAgent)
                    isProtected = chess_model.isProtected((row, col), board, agent)
                    if isThreatened:
                        safetyScore -= pieceValue * 2
                    if not isProtected:
                        safetyScore -= pieceValue

                    # Pawn Advancement: Collect pawn rows for dynamic reward
                    if piece == 12 and agent == "white":
                        pawnRows.append(row)
                    elif piece == 6 and agent == "black":
                        pawnRows.append(row)

                    # Center Control
                    if piece in pieceMultipliers:
                        centerControlReward += centerValues[row][col] * pieceMultipliers[piece]

                        # Additional reward for rooks on open files
                        if piece in [1, 7]:  # Rooks
                            if self.isOpenFile(col, board):
                                centerControlReward += 10

                        # Additional reward for bishops on long diagonals
                        if piece in [3, 9]:  # Bishops
                            if abs(row - col) in [0, 7] or (row + col == 7):
                                centerControlReward += 5

                    # Rook and Bishop Development
                    if piece in [1, 3, 7, 9]:
                        startingRow = 0 if agent == "black" else 7
                        if row != startingRow:
                            rookAndBishopDevelopment += 20

                    # Knight Development
                    if piece in [2, 8]:
                        if (row, col) in [(2, 1), (2, 6), (5, 1), (5, 6)]:
                            knightDevelopmentReward += 15

                    # Activation Bonus
                    if piece in [2, 3, 8, 9]:  # Knights and Bishops
                        startingRow = 0 if agent == "black" else 7
                        if row != startingRow:
                            activationBonus += 20
                    elif piece in [1, 4, 7, 10]:  # Rooks and Queen
                        startingRow = 0 if agent == "black" else 7
                        if row != startingRow:
                            activationBonus += 15

                    # Board Coverage
                    if piece in [1, 2, 3, 4, 7, 8, 9, 10]:
                        boardCoverageReward += abs(row - 3.5) + abs(col - 3.5)

                    # Living Queen Bonus
                    if piece in [4, 10]:
                        livingQueenBonus += 125

                elif piece in enemyPieces:
                    pieceValue = pieceValues[piece]
                    enemyMaterial += pieceValue

        # Game phase determination based on material
        if agentMaterial < 15 or enemyMaterial < 15:
            gamePhase = "endgame"
        elif agentMaterial > 30 and enemyMaterial > 30:
            gamePhase = "opening"
        else:
            gamePhase = "midgame"

        # Set weights depending on gamePhase
        if gamePhase == "opening":
            materialWeight = 4.5
            centerWeight = 2.0
            kingSafetyWeight = 1.5
            activationBonusWeight = 1.5
            pawnAdvancementWeight = 0.2
            safetyWeight = 0.9
            rookDevelopmentWeight = 1.0
            knightDevelopmentWeight = 1.0
            boardCoverageWeight = 0.5
            livingQueenWeight = 1.0
        elif gamePhase == "midgame":
            materialWeight = 4.0
            centerWeight = 1.5
            kingSafetyWeight = 2.0
            activationBonusWeight = 1.2
            pawnAdvancementWeight = 0.5
            safetyWeight = 1.3
            rookDevelopmentWeight = 1.0
            knightDevelopmentWeight = 1.0
            boardCoverageWeight = 0.5
            livingQueenWeight = 1.0
        else:  # Endgame
            materialWeight = 3.0
            centerWeight = 0.5
            kingSafetyWeight = 2.5
            activationBonusWeight = 2.0
            pawnAdvancementWeight = 3.0
            safetyWeight = 2.5
            rookDevelopmentWeight = 1.0
            knightDevelopmentWeight = 1.0
            boardCoverageWeight = 0.5
            livingQueenWeight = 1.0

        # Apply weights to respective components
        pawnAdvancementReward = 0  # Reset to recompute based on gamePhase
        if gamePhase in ["opening", "midgame"]:
            # Reward for advancing pawns towards the center
            for row in pawnRows:
                if agent == "white":
                    # Closer to promotion means higher row index
                    pawnAdvancementReward += row * 0.2
                else:
                    # Closer to promotion means lower row index
                    pawnAdvancementReward += (7 - row) * 0.2
        elif gamePhase == "endgame":
            # Reward based on proximity to promotion
            for row in pawnRows:
                if agent == "white":
                    # Distance to promotion row (7)
                    distance = 7 - row
                    pawnAdvancementReward += distance * 3.0  # High weight for proximity
                else:
                    # Distance to promotion row (0)
                    distance = row
                    pawnAdvancementReward += distance * 3.0  # High weight for proximity

        centerControlReward *= centerWeight
        rookAndBishopDevelopment *= rookDevelopmentWeight
        knightDevelopmentReward *= knightDevelopmentWeight
        livingQueenBonus *= livingQueenWeight
        boardCoverageReward *= boardCoverageWeight
        pawnAdvancementReward *= pawnAdvancementWeight
        activationBonus *= activationBonusWeight

        # Comprehensive Safety Evaluation
        safetyScore = self.safetyEvaluation(observation, agent) * safetyWeight

        # King Positioning Reward
        kingPositioningReward = self.kingPositioningReward(observation, agent) * kingSafetyWeight

        # King Safety Reward
        kingSafetyScore = self.kingSafetyEvaluation(observation, agent) * kingSafetyWeight

        # Capture Risk Evaluation
        captureRiskEvaluation = sum(
            self.evaluateCaptureRisk(move, observation, agent) for move in allActions
        )

        # Win/Loss Detection
        winLoss = (
            1 if (not any(chess_model.getActionMask(enemyAgent, observation)) and chess_model.inCheck(enemyAgent, board))
            else -1 if (not any(chess_model.getActionMask(agent, observation)) and chess_model.inCheck(agent, board))
            else 0
        )

        # Stalemate Penalty
        stalematePenalty = -5000 if chess_model.isStalemate(observation, enemyAgent) else 0

        # Final Evaluation Score
        evaluationScore = (
            (agentMaterial * materialWeight) +                 # Material advantage
            mobility +                                         # Number of valid moves available
            safetyScore +                                      # Safety evaluation
            kingSafetyScore +                                  # King Safety Evaluation
            livingQueenBonus +                                 # Incentive to keep the queen alive
            pawnAdvancementReward +                            # Reward for advancing pawns strategically or promoting
            centerControlReward +                              # Control over the center of the board
            captureRiskEvaluation +                            # Evaluating capture opportunities and risks
            rookAndBishopDevelopment +                         # Development of rooks and bishops
            knightDevelopmentReward +                          # Development of knights
            kingPositioningReward +                            # Positioning of the enemy king
            activationBonus +                                  # Encouragement for activating pieces
            boardCoverageReward +                              # Spread and activity of pieces on the board
            stalematePenalty +                                 # Penalty to discourage stalemates
            (winLoss * 15000)                                  # High score for checkmate or loss
        )

        return int(evaluationScore)

    def materialEvaluation(self, observation, agent):
        board = observation["board"]
        agentPieces = chess_model.PLAYER_PIECES[agent]

        agentScore = sum(self.getPieceValue(board[row][col], agent) 
                        for row in range(8) for col in range(8) 
                        if board[row][col] in agentPieces)

        return agentScore

    def getPieceValue(self, piece, agent):
        pieceValues = {
            0: 0,    # Empty square
            1: 6,    # Black Rook
            2: 3,    # Black Knight
            3: 3,    # Black Bishop
            4: 9,    # Black Queen
            5: 0,    # Black King
            6: 1,    # Black Pawn
            7: 6,    # White Rook 
            8: 3,    # White Knight
            9: 3,    # White Bishop
            10: 9,   # White Queen
            11: 0,   # White King
            12: 1    # White Pawn
        }
        return pieceValues[piece] if piece in chess_model.PLAYER_PIECES[agent] else -pieceValues[piece]

    def safetyEvaluation(self, observation, agent):
        board = observation["board"]
        enemyAgent = "white" if agent == "black" else "black"

        threatValues = {
            0: 0,    # Empty square
            1: 5,    # Black Rook
            2: 3,    # Black Knight
            3: 3,    # Black Bishop
            4: 20,   # Black Queen
            5: 0,    # Black King
            6: 1,    # Black Pawn
            7: 5,    # White Rook
            8: 3,    # White Knight
            9: 3,    # White Bishop
            10: 20,  # White Queen
            11: 0,   # White King
            12: 1    # White Pawn
        }

        safetyScore = 0
        enemyThreatReward = 0

        for row in range(8):
            for col in range(8):
                piece = board[row][col]

                if piece in chess_model.PLAYER_PIECES[agent]:
                    threatCount = chess_model.threatCount((row, col), board, enemyAgent)
                    isNearKing = chess_model.isPieceSafeFromKing([row, col], board, agent)
                    penaltyMultiplier = 2 if isNearKing and not chess_model.isProtected((row, col), board, agent) else 1
                    agentThreatPenalty = (threatCount ** 2) * threatValues[piece] * penaltyMultiplier

                    isThreatened = chess_model.isThreatened((row, col), board, enemyAgent)
                    isProtected = chess_model.isProtected((row, col), board, agent)

                    if isThreatened:
                        safetyScore -= self.getPieceValue(piece, agent) * 2
                    if not isProtected:
                        safetyScore -= self.getPieceValue(piece, agent)

                    safetyScore -= agentThreatPenalty * 1.5 

                elif piece in chess_model.PLAYER_PIECES[enemyAgent]:
                    threatCount = chess_model.threatCount((row, col), board, agent)
                    enemyThreatReward += (threatCount ** 1.5) * threatValues[piece] * 1.5

        # Combine enemy threat rewards
        safetyScore += enemyThreatReward

        return safetyScore

    def kingPositioningReward(self, observation, agent):
        # Encourage moving enemy king towards the corners or edges
        enemyAgent = "white" if agent == "black" else "black"
        enemyKingPosition = chess_model.findKingPosition(observation, enemyAgent)
        row, col = enemyKingPosition
        reward = 0

        # Reward based on proximity to corners and edges
        if row in [0, 7] and col in [0, 7]:  # Corner
            reward += 20
        elif row in [0, 7] or col in [0, 7]:  # Edge
            reward += 10
        return reward
    
    def kingSafetyEvaluation(self, observation, agent):
        board = observation["board"]
        enemyAgent = "white" if agent == "black" else "black"
        agentKingPos = chess_model.findKingPosition(observation, agent)
        if not agentKingPos:
            # If king position not found, assign a severe penalty (e.g., checkmate)
            return -10000

        row, col = agentKingPos
        safetyScore = 0

        # Kings Position Safety
        if (agent == "white" and row >= 6 and row <=7) or (agent == "black" and row >= 0 and row <=1):
            # King is towards the back
            safetyScore += 100
        else:
            # King is more exposed
            safetyScore -= 100

        # Pawn Shield Strength
        # Evaluate pawns in front of the king
        pawnShieldSquares = self.getPawnShieldSquares(agentKingPos, agent)
        pawnShieldCount = 0
        for pos in pawnShieldSquares:
            r, c = pos
            pawn = 6 if agent == "black" else 12  # Black Pawn or White Pawn
            if board[r][c] == pawn:
                pawnShieldCount += 1

        safetyScore += pawnShieldCount * 20  # Each pawn shield adds to safety

        # Count the number of opponent pieces threatening squares around the king
        threatenedSquares = self.getThreatenedSquares(agentKingPos, observation, enemyAgent)
        threatCount = len(threatenedSquares)
        safetyScore -= threatCount * 30  # Each threat reduces safety

        # King Mobility
        # Less mobility usually indicates better safety
        kingMoves = chess_model.getPieceMoves(observation, agentKingPos[0], agentKingPos[1])
        safetyScore += (8 - len(kingMoves)) * 10  # Fewer available moves can mean safer

        return safetyScore
    
    def getPawnShieldSquares(self, kingPos, agent):
        row, col = kingPos
        shieldSquares = []
        # Define the starting row based on the agent
        startingRow = 7 if agent == "white" else 0
        
        # Define a threshold to decide the side based on current king's column
        if agent == "white":
            if col >= 5:  # Likely kingside castled
                shieldSquares = [
                    (row-1, 5),  # f2
                    (row-1, 6),  # g2
                    (row-1, 7)   # h2
                ]
            else:  # Likely queenside castled
                shieldSquares = [
                    (row-1, 2),  # c2
                    (row-1, 3),  # d2
                    (row-1, 4)   # e2
                ]
        else:
            if col >= 5:  # Likely kingside castled
                shieldSquares = [
                    (row+1, 5),  # f7
                    (row+1, 6),  # g7
                    (row+1, 7)   # h7
                ]
            else:  # Likely queenside castled
                shieldSquares = [
                    (row+1, 2),  # c7
                    (row+1, 3),  # d7
                    (row+1, 4)   # e7
                ]
        
        # Ensure all shield squares are within the board
        shieldSquares = [
            (r, c) for r, c in shieldSquares 
            if 0 <= r < 8 and 0 <= c < 8
        ]
        
        return shieldSquares
    
    def getThreatenedSquares(self, kingPos, observation, enemyAgent):
        # Get all squares the enemy can attack
        enemyThreats = chess_model.getThreatenedPositions(observation, enemyAgent)
        # Filter threats that are adjacent to the king
        threatenedSquares = []
        for threat in enemyThreats:
            if abs(threat[0] - kingPos[0]) <=1 and abs(threat[1] - kingPos[1]) <=1:
                threatenedSquares.append(threat)
        return threatenedSquares

    def evaluateCaptureRisk(self, move, observation, agent):
        enemyAgent = "white" if agent == "black" else "black"
        targetPiece = chess_model.getTargetPiece(move, observation, agent)
        if targetPiece is None:
            return 0

        pieceValue = self.getPieceValue(targetPiece, agent)
        moveEndPosition = chess_model.getMoveEndPosition(move, agent)
        attackerValue = self.getPieceValue(chess_model.getStartPiece(move, observation, agent), agent)

        # Strongly prioritize captures involving promotions, especially if capturing a valuable piece
        promotionPiece = chess_model.getPromotionPiece(move, agent)
        if promotionPiece in [4, 10]:  # Queen promotion
            promotionReward = 300  # Base reward for promotion
            if pieceValue > attackerValue:  # Favorable trade on promotion
                promotionReward += pieceValue * 2
            return promotionReward

        # Discourage unfavorable trades and reward safe captures
        if chess_model.isCapture(move, observation, agent):
            # Check if the capturing piece will be under threat after the move
            isCapturingPieceSafe = not chess_model.isThreatened(moveEndPosition, observation["board"], enemyAgent)

            if isCapturingPieceSafe:
                # Reward for favorable trades and safe captures
                tradeBonus = (pieceValue - attackerValue) * 100 if pieceValue > attackerValue else 0  # Increased bonus
                return (pieceValue * 2.0) + tradeBonus
            else:
                # Penalize for leaving the capturing piece vulnerable
                penalty = self.getPieceValue(chess_model.getStartPiece(move, observation, agent), agent) * 2  # Double penalty
                return -penalty

        return 0

    def isOpenFile(self, col, board):
        # Check if the column is open (no pawns blocking)
        for row in range(8):
            piece = board[row][col]
            if piece in [6, 12]:  # Pawns
                return False
        return True

    def isPassedPawn(self, position, board, agent):
        # Determine if a pawn is passed (no opposing pawns blocking)
        row, col = position
        enemyAgent = "white" if agent == "black" else "black"
        enemyPawn = 12 if enemyAgent == "white" else 6
        direction = -1 if agent == "white" else 1
        for r in range(row + direction, 8 if agent == "black" else -1, direction):
            for c in [col - 1, col, col + 1]:
                if 0 <= c <= 7 and board[r][c] == enemyPawn:
                    return False
        return True

    def isInEnemyTerritory(self, row, agent):
        # Check if a position is in the enemy's half of the board
        return (row >= 4 and agent == "white") or (row <= 3 and agent == "black")

    def isCloseToEnemyKing(self, position, observation, agent):
        # Check if a position is close to the enemy king
        enemyAgent = "white" if agent == "black" else "black"
        enemyKingPosition = chess_model.findKingPosition(observation, enemyAgent)
        distance = abs(position[0] - enemyKingPosition[0]) + abs(position[1] - enemyKingPosition[1])
        return distance <= 2

    def isThreateningHigherValuePiece(self, move, observation, agent):
        # Check if the move threatens an enemy piece of higher value
        enemyAgent = "white" if agent == "black" else "black"
        nextObservation = chess_model.RESULTS(observation, move, agent)
        enemyPieces = chess_model.PLAYER_PIECES[enemyAgent]
        startPiece = chess_model.getStartPiece(move, observation, agent)
        threats = chess_model.getThreatenedPositions(nextObservation, agent)
        for pos in threats:
            piece = nextObservation["board"][pos[0]][pos[1]]
            if piece in enemyPieces and self.getPieceValue(piece, enemyAgent) > self.getPieceValue(startPiece, agent):
                return True
        return False

    def isMoveSafe(self, move, observation, agent):
        nextObservation = chess_model.RESULTS(observation, move, agent)
        kingPos = chess_model.findKingPosition(nextObservation, agent)
        if kingPos is None:
            # If the king is missing, it's either checkmate or a win condition.
            return False
        return not chess_model.isThreatened(kingPos, nextObservation["board"], "white" if agent == "black" else "black")
    
    def isAdvantageousCheck(self, move, observation, agent):
        # Simulate the move
        nextObservation = chess_model.RESULTS(observation, move, agent)
        
        # Check for checkmate
        if chess_model.isCheckmate(move, nextObservation, agent):
            return True  # Immediate win
        
        # Check for material gain
        targetPiece = chess_model.getTargetPiece(move, observation, agent)
        if targetPiece is not None:
            pieceValue = self.getPieceValue(targetPiece, agent)
            if pieceValue > 0:  # Assuming positive value indicates valuable piece
                return True  # Capturing a valuable piece
        
        return False  # No immediate advantage
    
    def findQueenPosition(self, observation, agent):
        board = observation["board"]
        queenPiece = 4 if agent == "black" else 10  # Black Queen: 4, White Queen: 10
        for row in range(8):
            for col in range(8):
                if board[row][col] == queenPiece:
                    return (row, col)
        return None