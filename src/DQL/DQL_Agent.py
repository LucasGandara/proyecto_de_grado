#!/usr/bin/env python   
# -*- coding: utf-8 -*-

from Robot_env import Robot_env
from collections import deque
import numpy as np
import random
from tqdm import tqdm
from Model import Sequential

# Some DQL values
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class DQNAgent:
    def __init__(self):
        #Main model
        self.model = Sequential([369, 1], n_out=9, ini_type='xavier')
        
        # Target model
        self.target_model = Sequential([369, 1], n_out=9, ini_type='xavier', from_model=True, model=self.model)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, model='model'):
        if model == "model":
            return self.model.forward_prop(np.array([state]).T)
        if model == 'target':
            return self.target_model.forward_prop(np.array([state]).T)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = np.array([self.get_qs(state) for state in current_states])
        
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = np.array([self.get_qs(state, 'target') for state in new_current_states])
        
        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)

        # Here we train the NN
        self.model.train(np.array(X), np.array(Y), LR=0.001, N_Epochs=1)

        #Updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model = Sequential([369, 1], n_out=9, ini_type='xavier', from_model=True, model=self.model)
            self.target_update_counter = 0