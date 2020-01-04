import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sokoban import Game

import tensorflow as tf
from collections import deque
import random

class Agent:
    def __init__(self, environment, optimizer):
        # Initialize atributes

        self._action_size = Game.ACTION_SPACE_SIZE
        self._optimizer = optimizer
        
        self.experience_replay = deque(maxlen=10000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()
        

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(10,11,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(128, (3, 3)))
        # model.add(Activation('relu'))
        # # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(256, (3, 3)))
        # model.add(Activation('relu'))
        # # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(256, (4, 4)))
        # model.add(Activation('relu'))
        # # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # this converts our 3D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(Game.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=self._optimizer, metrics=['accuracy'])
        model.summary()
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0,Game.ACTION_SPACE_SIZE-1)
        
        q_values = self.q_network.predict(state.reshape((1,10,11,1)))
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state.reshape((1,10,11,1)))
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state.reshape((1,10,11,1)))
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state.reshape((1,10,11,1)), target, epochs=1, verbose=0)