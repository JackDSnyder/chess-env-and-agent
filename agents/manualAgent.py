import numpy as np
import chess_environment.env.chess_model as chess_model

class Agent:

    def __init__(self):
        return
    
    def reset(self):
        return
    
    # Returns action or -1 for invalid action
    def notationToAction(self, move, agent):
        # Remove quality checks
        move = move.replace("+", "").replace("#", "").replace("!", "").replace("?", "").replace("x", "")
        
        colToIndex = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        rowToIndex = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

        # Castling:
        if move == "0-0": # kingside castling
            if agent == "white":
                return chess_model.maskIndex(7, 4, 7, 6, -1)
            else:
                return chess_model.maskIndex(0, 4, 0, 6, -1)
        elif move == "0-0-0": # queenside castling
            if agent == "white":
                return chess_model.maskIndex(7, 4, 7, 2, -1)
            else:
                return chess_model.maskIndex(0, 4, 0, 2, -1)
            
        # Promotion
        if len(move) == 5 and move[-1] in "QRBN":
            promotionMap = {'Q': 0, 'R': 1, 'B': 2, 'N': 3}
            sc = colToIndex[move[0]]
            sr = rowToIndex[move[1]]
            ec = colToIndex[move[2]]
            er = rowToIndex[move[3]]
            p = promotionMap[move[4]]
            print(f"Promotion move parsed: sr={sr}, sc={sc}, er={er}, ec={ec}, promotion={p}")
            return chess_model.maskIndex(sr, sc, er, ec, p)
        
        # Regular moves
        if len(move) == 4:
            sc = colToIndex[move[0]]
            sr = rowToIndex[move[1]]
            ec = colToIndex[move[2]]
            er = rowToIndex[move[3]]
            # print(f"Regular move parsed: sr={sr}, sc={sc}, er={er}, ec={ec}")
            return chess_model.maskIndex(sr, sc, er, ec, None)

        # Invalid move
        print("Invalid move notation")
        return -1


    
    def agent_function(self, observation, agent):
        action_mask = observation["action_mask"]
        while True:
            print()
            print("+-- Move Notation --+")
            print("Regular move: e4e5 - move e4 piece to e5")
            print("Promotion: e2e1Q - move e2 pawn to e1 and promote to queen, can use QRBN for promotion piece")
            print("Kingside castle: 0-0, Queenside castle: 0-0-0")
            move = input("Input a move: ")
            action = self.notationToAction(move, agent)
            if action != -1 and action_mask[action]:
                chess_model.printMove(action, agent)
                return action
            print(f"Action: {action}, mask[action]: {action_mask[action]}")
            print("Invalid notation.")
