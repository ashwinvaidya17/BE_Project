from grabscreen import grab_screen
import tensorflow as tf
from tensorflow import keras as keras
from directKeys import PressKey, ReleaseKey, SPACE
import cv2
import numpy as np
import random
import time
from collections import deque
from flappybirdTry import FlappyBird
import pygame
import os
import argparse


parser = argparse.ArgumentParser(description='RL Flappy bird')
parser.add_argument('-d','--display', help='Display frames', required=False, default='False')
parser.add_argument('-v','--verbose', help='Displays additional data', required=False, default='False')
args = vars(parser.parse_args())

class Agent:
    def __init__(self):
        self.gamma = 0.9
        self.episodes = 1000
        self.framesPerEpisode = 10000
        self.epsilon = 1.0
        self.minEpsilon = 0.1
        self.epsilonDecayRate = 0.01
        self.episodeMoves = deque(maxlen=10000)
        self.framesStacked = 4
        if args['display'] == 'True':
            self.display = True
        else:
            self.display = False
        if args['verbose'] == 'True':
            self.verbose = True
        else:
            self.verbose = False
        self.model = self.createModel()


    def createModel(self):
        if os.path.isfile('agent_ConvLSTM2d.h5py'):
            model = keras.models.load_model('agent_ConvLSTM2d.h5py')
        else:
            model = keras.models.Sequential()
            model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    input_shape=(self.framesStacked, 1, 40, 70),
                    padding='same', return_sequences=True, data_format='channels_first'))
                
            model.add(keras.layers.BatchNormalization())

            model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                    padding='same', return_sequences=True))
            model.add(keras.layers.BatchNormalization())

            model.add(keras.layers.Flatten())

            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense(2, activation='linear'))

            model.compile(loss='mse', optimizer=keras.optimizers.Adam())

        return model

    def preprocess(self, screen):
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (40, 70))
        screen = screen.reshape(40, 70, 1)
        return screen

    def act(self, frame, game):
        if np.random.rand() < self.epsilon:
            action = random.randrange(2)
        else:
            action = np.argmax(self.model.predict(np.array(frame).reshape(-1, 4, 1, 40, 70)))

        if action == 1:
            # Jump
            game.jumpAction()
            # keyDown('space')
            # PressKey(SPACE)
            # time.sleep(0.05)
            # ReleaseKey(SPACE)
        return action

    def play(self):
        try:
            pygame.font.init()
            screens = deque(maxlen=self.framesStacked)
            nextScreens = deque(maxlen=self.framesStacked)
            font = pygame.font.SysFont("Arial", 50)
            for episode in range(1, self.episodes):
                game = FlappyBird(self.display)
                clock = pygame.time.Clock()
                frameCount = 0
                rewardPerFrame = 1
                penalty = -100
                isAlive, screen = game.step(clock, font)
                screen = self.preprocess(screen)
                for _ in range(0, self.framesStacked):
                    screens.append(screen)
                    nextScreens.append(screen)
                # screen = grab_screen(region=(0, 30, 400, 738))
                frameCount = frameCount + 1

                while True:
                    if frameCount == self.framesPerEpisode or isAlive == False:
                        break
                    
                    action = self.act(screens, game)
                    initialScore = game.counter
                    isAlive, nextScreen = game.step(clock, font)
                    # ReleaseKey(SPACE)
                    # To skip frames
                    # if action == 1:
                    #     for i in range(0, 3):
                    #         isAlive = game.step(clock, font)

                    # nextScreen = grab_screen(region=(0, 30, 400, 738))
                    nextScreen = self.preprocess(nextScreen)
                    nextScreens.append(nextScreen)
                    finalScore = game.counter
                    if isAlive:
                        reward = rewardPerFrame
                    elif finalScore > initialScore:
                        reward = 100
                    else:
                        reward = penalty
                    screen = nextScreen
                    self.episodeMoves.append([np.array(screens), np.array(nextScreens), reward, isAlive, action])
                    if self.verbose:
                        print("Action: {}, Epsilon: {}, reward: {}".format(action, self.epsilon, reward))
                    for i in range(0, self.framesStacked):
                        screens[i] = nextScreens[i]
                    frameCount = frameCount + 1


                print("Episode #{}, Score: {}, Frames: {}".format(episode, finalScore, frameCount))
                self.train()
                if episode % 40 == 0:
                    self.model.save('agent_ConvLSTM2d.h5py')

        except KeyboardInterrupt:
            self.model.save('agent_ConvLSTM2d.h5py')


    def train(self):
        batchSize = 1000
        if batchSize > len(self.episodeMoves):
            batchSize = len(self.episodeMoves)
        
        minibatch = random.sample(self.episodeMoves, batchSize)
        for batch in minibatch:
            state = batch[0].reshape(-1, 4, 1, 40, 70)
            nextState = batch[1].reshape(-1, 4, 1, 40, 70)
            reward = batch[2]
            isDone = batch[3]
            action = batch[4]

            target = reward     # in case of penalty
            if not isDone:
                target = reward + np.amax(self.model.predict(nextState)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # print("target: {}\nlen(target_f): {}\ntarget_f: {}".format(target, len(target_f), target_f))

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.minEpsilon:
            self.epsilon = self.epsilon - self.epsilonDecayRate


if __name__ == '__main__':
    Agent().play()