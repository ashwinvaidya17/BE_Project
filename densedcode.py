import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras import layers
# from tensorflow.python.keras.layers import Input, Dense
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras import Model
from numpy import newaxis
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mlagents.envs import UnityEnvironment
# from tensorflow.python.keras.callbacks import TensorBoard
# import cv2
import os
from collections import deque

epsilon = 0.2 # used to clip
entfact = 2e-2 # entropy factor, to encourage exploration
WIDTH = 100
HEIGHT = 40
NUM_FRAMES = 7

def PPO_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        denom = 1.772
        # prob = y_true * y_pred # maybe need K.sum
        # old_prob = y_true* old_prediction
        prob = keras.backend.exp(-1 * keras.backend.square(y_true - y_pred) / 2) / denom
        old_prob = keras.backend.exp(-1 * keras.backend.square(y_true - old_prediction) / 2) / denom
        r = prob/(old_prob + 1e-10)

        return -keras.backend.mean(keras.backend.minimum(r* advantage, keras.backend.clip(r, min_value=1 - epsilon, max_value= 1 + epsilon)*advantage) + entfact *(
                       prob *keras.backend.log(prob + 1e-10)))
    return loss

class PPOAgent:
    def __init__(self, env, n_actions, n_features, action_low=-1, action_high=1, reward_decay=0.99,
                 actor_learning_rate=0.01, critic_learning_rate=0.01, learning_rate_decay=0.95):
        self.env = env
        self.state_size = n_features
        self.action_size = n_actions
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = reward_decay   # discount rate
        self.vector_observation_shape = [1, 1, NUM_FRAMES]
        # self.visual_observation_shape = [16, 40, 1]
        self.visual_observation_shape = [WIDTH, HEIGHT, NUM_FRAMES]
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate # often larger than actor_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = 64
        self.lam = 0.95 # gae factor
        self.memory = [] # store (s, a, r) for one agent
        self.agents = 1 # number of agents that collect memory | eventhough this is defined, autocar uses only 1 agent
        self.history = {} # store the memory for different agents
        self.history['states'] = []
        self.history['observations'] = []
        self.history['actions'] = []
        self.history['discounted_rs'] = []
        self.history['advantages'] = []
        self.DUMMY_ADVANTAGE = np.zeros((1,1))
        self.DUMMY_PREDICTION = np.zeros((1,self.action_size))
        self.actor_model = self.actor_network()
        self.actor_old_model = self.actor_network()
        self.actor_old_model.set_weights(self.actor_model.get_weights())
        self.critic_model= self.critic_network()
        self.writer = keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=False)
    

    def load_weights(self, path):
        self.actor_model.load_weights(path+'_actor.h5')
        self.actor_old_model.load_weights(path+'_actor.h5')
        self.critic_model.load_weights(path + '_critic.h5')

    def save_weights(self, path):
        self.actor_model.save_weights(path+'_actor.h5')
        self.critic_model.save_weights(path + '_critic.h5')

    def actor_network(self):
        in1 = keras.layers.Input(shape=self.visual_observation_shape, name="visual_state")
        conv_layer1 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(in1)
        norm1 = keras.layers.BatchNormalization()(conv_layer1)
        conv_layer2 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(norm1)
        norm2 = keras.layers.BatchNormalization()(conv_layer2)
        conv_layer3 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(norm2)
        norm3 = keras.layers.BatchNormalization()(conv_layer3)
        conv_layer4 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(norm3)
        norm4 = keras.layers.BatchNormalization()(conv_layer4)
        in1flat = keras.layers.Flatten()(norm4)
        dense1 = keras.layers.Dense(1024, activation='relu')(in1flat)
        dropout1 = keras.layers.Dropout(0.4)(dense1)
        dense2 = keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = keras.layers.Dropout(0.4)(dense2)
        dense3 = keras.layers.Dense(64, activation='relu')(dropout2)
        dropout3 = keras.layers.Dropout(0.4)(dense3)
        dense4 = keras.layers.Dense(16, activation='relu')(dropout3)
        in2 = keras.layers.Input(shape=(1, NUM_FRAMES), name="vector_state")
        in2_flat = keras.layers.Flatten()(in2)
        dense_in2_1 = keras.layers.Dense(16)(in2_flat)
        l1 = keras.layers.concatenate([dense4, dense_in2_1])
        l4 = keras.layers.Dense(32, activation='relu')(l1)
        action = keras.layers.Dense(self.action_size, activation="tanh")(l4)
        advantage = keras.Input(shape=(1,), name='advantage')
        old_prediction = keras.Input(shape=(self.action_size,), name='old_predction')
        actor = keras.Model(inputs=[in1,in2, advantage, old_prediction], outputs=action)
        actor.compile(optimizer=keras.optimizers.Adam(lr=self.actor_learning_rate), loss=[PPO_loss(advantage=advantage,old_prediction=old_prediction)])
        return actor
       
    def critic_network(self):
        in1 = keras.layers.Input(shape=self.visual_observation_shape, name="visual_state")
        conv_layer1 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last')(in1)
        norm1 = keras.layers.BatchNormalization()(conv_layer1)
        conv_layer2 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(norm1)
        norm2 = keras.layers.BatchNormalization()(conv_layer2)
        conv_layer3 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(norm2)
        norm3 = keras.layers.BatchNormalization()(conv_layer3)
        conv_layer4 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                    input_shape=self.visual_observation_shape,
                    padding='same', data_format='channels_last', activation='relu')(norm3)
        norm4 = keras.layers.BatchNormalization()(conv_layer4)
        in1flat = keras.layers.Flatten()(norm4)
        dense1 = keras.layers.Dense(1024, activation='relu')(in1flat)
        dropout1 = keras.layers.Dropout(0.4)(dense1)
        dense2 = keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = keras.layers.Dropout(0.4)(dense2)
        dense3 = keras.layers.Dense(64, activation='relu')(dropout2)
        dropout3 = keras.layers.Dropout(0.4)(dense3)
        dense4 = keras.layers.Dense(16, activation='relu')(dropout3)
        in2 = keras.layers.Input(shape=(1, NUM_FRAMES), name="vector_state")
        in2_flat = keras.layers.Flatten()(in2)
        dense_in2_1 = keras.layers.Dense(16)(in2_flat)
        l1 = keras.layers.concatenate([dense4, dense_in2_1])
        l4 = keras.layers.Dense(32, activation='relu')(l1)
        value = keras.layers.Dense(1)(l4)
        critic = keras.Model(inputs=[in1, in2], outputs=value)
        critic.compile(optimizer=keras.optimizers.Adam(lr=self.critic_learning_rate), loss='mse')
        return critic

    def choose_action(self, vector_observation, visual_observation, train=True):
        a = self.actor_model.predict([visual_observation, vector_observation, self.DUMMY_ADVANTAGE, self.DUMMY_PREDICTION])
        return np.clip(a, self.action_low, self.action_high)

    def remember(self, vector_observation, visual_observation, action, reward, next_vector_observation, next_visual_observation):
        self.memory += [[vector_observation, visual_observation, action, reward, next_vector_observation, next_visual_observation]]

    def discount_rewards(self, rewards, gamma, value_next=0.0):
        discounted_r = np.zeros_like(rewards)
        running_add = value_next
        for t in reversed(range(0, len(rewards))):
            discounted_r[t] = running_add = running_add * gamma + rewards[t]
        return discounted_r
    
    def process_memory(self):
        memory = np.vstack(np.array(self.memory))
        vector_states = np.vstack(memory[:,0])
        visual_observations = np.vstack(memory[:,1]).reshape((-1, WIDTH, HEIGHT, NUM_FRAMES))
        actions = np.vstack(memory[:,2])
        rewards = memory[:,3]
        discounted_episode_rewards = self.discount_rewards(rewards, self.gamma)[:,newaxis]
        value_estimates = self.critic_model.predict([visual_observations,vector_states])
        value_estimates = np.append(value_estimates, 0)
        delta_t = rewards + self.gamma * value_estimates[1:] - value_estimates[:1] # eq 12 PPO paper
        advantages = self.discount_rewards(delta_t, self.gamma * self.lam)[:,newaxis] # eq 11 PPO paper
        last = vector_states.shape[0]
        self.history['states'] += [vector_states[-last:]]
        self.history['observations'] += [visual_observations[-last:]]
        self.history['actions'] += [actions[-last:]]
        self.history['discounted_rs'] += [discounted_episode_rewards[-last:]]
        self.history['advantages'] += [advantages[-last:]]
        self.memory = [] # empty the memory

    def replay(self):
        vector_observations = np.vstack(self.history['states'])
        visual_observations = np.vstack(self.history['observations'])
        actions = np.vstack(self.history['actions'])
        discounted_rewards = np.vstack(self.history['discounted_rs'])
        advantages = np.vstack(self.history['advantages'])
        old_prediction = self.actor_old_model.predict_on_batch([visual_observations, vector_observations, self.DUMMY_ADVANTAGE, self.DUMMY_PREDICTION])
        actor_loss = self.actor_model.fit([visual_observations, vector_observations,advantages, old_prediction],[actions], batch_size=self.batch_size, shuffle=True, epochs=10, verbose=False, callbacks=[self.writer])
        critic_loss = self.critic_model.fit([visual_observations, vector_observations],[discounted_rewards], batch_size=self.batch_size, shuffle=True, epochs=10, verbose=False) # fit with discounted rewards as per autocar

        for key in self.history:
            self.history[key] = []
        
        self.actor_old_model.set_weights(self.actor_model.get_weights())
        
#-------------------------------------------------------------------------------------
env_name = 'Linux'
#file_name=env_name
env = UnityEnvironment(file_name=env_name)

# Examine environment parameters
print(str(env))

# Set the default brain to work with
default_brain = env.brain_names[1]
brain = env.brains[default_brain]


# Reset the environment
#env_info = env.reset(train_mode=False)[default_brain]
env_info = env.reset()
brainInfo = env_info['CarLearning']
print(np.array(brainInfo.visual_observations).shape)

agent = PPOAgent(env,
                n_actions=2,
                n_features=1,
                actor_learning_rate=1e-5,
                critic_learning_rate=2e-5
                )
rewards = []

# PPO
n_episodes = 100000

if os.path.exists('./model/PPOKeras_actor.h5'):
    agent.load_weights('./model/PPOKeras')
    print("loading existing model")
else:
    print("no existing model found")

current_vectors = deque(maxlen=NUM_FRAMES)
next_vectors = deque(maxlen=NUM_FRAMES)
current_visuals = deque(maxlen=NUM_FRAMES)
next_visuals = deque(maxlen=NUM_FRAMES)

for i_episode in range(n_episodes):
    env_info = agent.env.reset(train_mode=True)[default_brain]
    state = env_info.vector_observations[0][0]
    observation = env_info.visual_observations[0][0].reshape(WIDTH, HEIGHT)
    current_vectors.clear()
    next_vectors.clear()
    current_visuals.clear()
    next_visuals.clear()

    for _ in range(0, NUM_FRAMES):
        current_visuals.append(observation)
        next_visuals.append(observation)
        current_vectors.append(state)
        next_vectors.append(state)
    r = 0
    
    while True:
        action = agent.choose_action(np.array(current_vectors).reshape(*agent.vector_observation_shape), np.array(current_visuals).reshape(-1, WIDTH, HEIGHT, NUM_FRAMES))
        env_info = agent.env.step(action)[default_brain]
        next_state = env_info.vector_observations[0][0]
        next_observation = env_info.visual_observations[0][0].reshape(WIDTH, HEIGHT)
        next_vectors.append(next_state)
        next_visuals.append(next_observation)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        r += reward
        agent.remember(np.array(current_vectors).reshape(*agent.vector_observation_shape), np.array(current_visuals), action, reward, np.array(next_vectors), np.array(next_visuals))

        for i in range(0, NUM_FRAMES):
            current_vectors[i] = next_vectors[i]
            current_visuals[i] = next_visuals[i]

        if done:
            print("episode:", i_episode+1, "rewards: %.2f" % r, end="\r")
            agent.process_memory()
            rewards += [r]
            break
    if (i_episode+1) % agent.agents == 0: # update every n_agent episodes
        agent.replay()
    if (i_episode+1) % 5 == 0:
        print("Saving weights #{}".format(i_episode))
        agent.save_weights('./model/PPOKeras')
print("\n")
print("finished learning!")
agent.save_weights('./model/PPOKeras')
env.close()
