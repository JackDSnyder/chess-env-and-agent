#!/usr/bin/env python3
import functools
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
import os
import time

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import chess_environment.env.chess_model as chess_model

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    
    metadata = {"render_modes": ["human"], "name": "chess_v0", 
                "is_parallelizable": False, "render_fps": 5,}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode
        """
        self.board = chess_model.getNewBoard()
        self.agents = ["white", "black"]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.action_spaces = {name: spaces.Discrete(4130) for name in self.agents}
        self.observation_spaces = {
            name: spaces.Dict({
                "observation": spaces.Dict({
                    "board": spaces.Box(
                        low=0, high=12, shape=(8, 8), dtype=np.int8  # 8x8 grid with values 0-12 for pieces and empty spaces
                    ),
                    "kingMoved": spaces.Dict({
                        "white": spaces.Discrete(2),  # 0 or 1 to represent False/True
                        "black": spaces.Discrete(2)
                    }),
                    "rookMoved": spaces.Dict({
                        "whiteKingside": spaces.Discrete(2),
                        "whiteQueenside": spaces.Discrete(2),
                        "blackKingside": spaces.Discrete(2),
                        "blackQueenside": spaces.Discrete(2)
                    }),
                    "enPassantLocation": spaces.Box(
                        low=0, high=7, shape=(2,), dtype=np.int8  # Row and column coordinates, or default [0, 0] if empty
                    )
                }),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(4130,), dtype=np.int8  # Represents valid actions with 0s and 1s
                )
            })
            for name in self.agents
        }


        self.rewards = None
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}

        self.agent_selection = None

        self.gameState = chess_model.GameState()
        self.fiftyRule = 0

        ### Pygame Stuff ###
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screenHeight = self.screenWidth = 400

        self.screen = None

        if self.render_mode == "human":
            self.BOARD_SIZE = (self.screenHeight, self.screenWidth)
            self.clock = pygame.time.Clock()
            self.cellSize = (self.BOARD_SIZE[0] / 8, self.BOARD_SIZE[1] / 8)
    

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        boardCopy = [row[:] for row in self.board]
        observation = {"board": boardCopy, 
                       "kingMoved": self.gameState.kingMoved,
                       "rookMoved": self.gameState.rookMoved,
                       "enPassantLocation": self.gameState.enPassantLocation}
        action_mask = chess_model.getActionMask(agent, observation)
        return {"observation": observation, "action_mask": action_mask}

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.board = chess_model.getNewBoard()
        self.agents = self.possible_agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}

        self.gameState = chess_model.GameState()
        self.fiftyRule = 0

        ### Pygame Stuff ###
        if self.render_mode == "human":
            self.render()

    def applyResult(self, result):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result * result_coef
            self.infos[name] = {"legal_moves": []}

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        - gameState class
        And any internal state used by observe() or render()
        """
        # handles stepping an agent which is already dead
        # accepts a None action for the one agent, and moves the agent_selection to
        # the next dead agent,  or if there are no more dead agents, to the next live agent
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            time.sleep(10)
            return self._was_dead_step(action)


        currentAgent = self.agent_selection
        enemyAgent = "black" if currentAgent == "white" else "white"
        
        # Verify action
        action = int(action)
        baseObservation = self.observe(currentAgent)
        observation, action_mask = baseObservation["observation"], baseObservation["action_mask"]
        assert action_mask[action] == 1
        # Apply action to environment 
        pieceStart, pieceEnd, specialMove = chess_model.actionToPositions(action, currentAgent)
        self.board, self.gameState, resetFifty = chess_model.makeMove(observation, currentAgent, pieceStart, pieceEnd, specialMove)
        self.fiftyRule = 0 if resetFifty else self.fiftyRule + 1

        enemyActionMask = self.observe(enemyAgent)["action_mask"]
        enemyMoveCount = 0
        for action in enemyActionMask:
            if action:
                enemyMoveCount += 1
        
        # Check for game over.
        # if enemyMoveCount is 0, it is stale or checkmate
        # if no capture has been made or pawn been moved in 50 moves, forced draw
        # if there is insufficient material, forced draw
        gameOver = enemyMoveCount == 0 or self.fiftyRule == 50 or chess_model.insufficientMaterial(self.board)
        if gameOver:
            print()
            chess_model.printBoard(self.board)
            if enemyMoveCount == 0:
                if chess_model.inCheck(enemyAgent, self.board):
                    # Checkmate
                    print(f"Checkmate - {currentAgent} wins!")
                    result = 1 if currentAgent == "white" else -1
                else:
                    # Stalemate
                    print("Stalemate")
                    result = 0
            else:
                # Draw
                print("Draw")
                result = 0
            self.applyResult(result)


        self._accumulate_rewards()

        self.agent_selection = (
            self._agent_selector.next()
        )

        ### Pygame stuff ###
        if self.render_mode == "human":
            self.render()

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "human":
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(self.BOARD_SIZE)
                pygame.display.set_caption("Chess")
                self.lightColor = (112, 102, 119)
                self.darkColor = (204, 183, 174)
                self.textColor = (0, 0, 0)
                self.fontSize = int(self.screenHeight / 8 * 0.8)  # Adjust font size based on cell size
                self.labelFont = pygame.font.SysFont("Arial", int(self.cellSize[0] * 0.3))  # Smaller font for labels
                basePath = os.path.dirname(__file__)
                self.font = pygame.font.Font(basePath + "/symbols.ttf", 64)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        # Draw board squares and pieces
        for row in range(8):
            for col in range(8):
                # Draw square
                color = self.darkColor if (row + col) % 2 == 0 else self.lightColor
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.cellSize[0], row * self.cellSize[1], self.cellSize[0], self.cellSize[1]))

                # Draw piece if present
                pieceCode = self.board[row][col]
                if pieceCode != 0:
                    pieceText = chess_model.UNICODE_MAPPING[pieceCode]
                    pieceSurface = self.font.render(pieceText, True, self.textColor)
                    pieceRect = pieceSurface.get_rect(center=(col * self.cellSize[0] + self.cellSize[0] / 2,
                                                                row * self.cellSize[1] + self.cellSize[1] / 2))
                    self.screen.blit(pieceSurface, pieceRect)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])


