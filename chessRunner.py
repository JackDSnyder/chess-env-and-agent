import agents.alphabetaAgent
import agents.manualAgent
import agents.random_agent
import chess_environment.chess_environment_v0 as myChessLibrary
import time
import agents

def createEnvironment(render_mode, seed=None):
    env = myChessLibrary.env(render_mode=render_mode)
    if seed:
        env.reset(seed)
    return env

def runEpisode(env, agent1, agent2):
    gameT1 = time.time()
    agents = { "white": agent1, "black": agent2 }
    times = { "white": 0.0, "black": 0.0 }
    env.reset()
    agent1.reset()
    agent2.reset()
    turns = 0

    for agent in env.agent_iter():
        turns += 1
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            break

        t1 = time.time()
        action = agents[agent].agent_function(observation, agent)
        t2 = time.time()
        times[agent] += (t2-t1)

        env.step(action)

    winner = None
    if len(env.rewards.keys()) == 2:
        for a in env.rewards:
            if env.rewards[a] == 1:
                winner = a
                break
    

    for agent in times:
        avgTime = times[agent] / turns*2
        print(f"{agent} took {avgTime:.5f} seconds per move.")
    gameTime = time.time() - gameT1
    print(f"Game time: {gameTime:8.5f} seconds.")

    return winner

def runManyEpisodes(env, episode_count, agent1, agent2):
    winners = {}
    for i in range(episode_count):
        winner = runEpisode(env, agent1, agent2)
        if winner not in winners:
            winners[winner] = 0
        winners[winner] += 1
    env.close()
    if 'player_0' not in winners.keys():
        winners['player_0'] = 0
    if 'player_1' not in winners.keys():
        winners['player_1'] = 0
    return winners



def main():
    agent1 = agents.alphabetaAgent.Agent()
    # agent2 = agents.manualAgent.Agent()
    agent2 = agents.random_agent.Agent()
    # agent2 = agents.random_agent.Agent()
    env = createEnvironment("human")
    gameType = input("Input 'W' to watch a run, 'M' to manually control agent 2, or 'T' to run games until error. ").upper()
    if gameType == "W":
        runEpisode(env, agent1, agent2)
    elif gameType == "M":
        agent2 = agents.manualAgent.Agent()
        runEpisode(env, agent1, agent2)
    elif gameType == "T":
            while True:
                runEpisode(env, agent1, agent2)

if __name__ == "__main__":
    main()
    