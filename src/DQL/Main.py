import numpy as np
from DQL_Agent import DQNAgent
from Robot_env import Robot_env
import random
from tqdm import tqdm

env = Robot_env()
agent = DQNAgent()

# Environment settings
EPISODES = 20000
SHOW_PREVIEW = False

# Exploration settings
epsilon = 1  # Not constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
AddCostEvery = 500

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='Episode'):
    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
    
        new_state, reward, done = env.step(action)

        if SHOW_PREVIEW:
            env.render()
        
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    
    if episode == EPISODES - 2:
        SHOW_PREVIEW = True

    if (step % AddCostEvery) == 0:
        cost.appen(agent.model.cost)