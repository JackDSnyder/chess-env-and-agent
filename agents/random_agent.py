import chess_environment.env.chess_model as chess_model
import numpy as np

class Agent:

    def __init__(self):
        return
    
    def reset(self):
        return
    
    def agent_function(self, observation, agent):
        action = np.random.choice(np.where(observation['action_mask'])[0])
        chess_model.printMove(action, agent)
        return action