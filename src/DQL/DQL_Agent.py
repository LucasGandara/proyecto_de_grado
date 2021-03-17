#!/usr/bin/env python   
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

from Robot_env import Robot_env
from collections import deque
import numpy as np
import random
from tqdm import tqdm
import pickle as pkl

# Some DQL values
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class DQNAgent:
    def __init__(self):
        #Main model
        self.model = self.create_model()
        
        # Target model
        self.target_model = self.create_model()

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, model='model'):
        return self.model.predict(state.reshape(-1, * state.shape) / 255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        
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
       # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        #Updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
    def create_model(self):
        modelo = keras.Sequential()
        modelo.add(keras.layers.Dense(361, activation='linear'))
        modelo.add(keras.layers.Dropout(0.2))


        modelo.add(keras.layers.Dense(units=64, activation='linear'))
        modelo.add(keras.layers.Dropout(0.2))

        modelo.add(keras.layers.Dense(9, activation="linear"))
        modelo.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

        return modelo