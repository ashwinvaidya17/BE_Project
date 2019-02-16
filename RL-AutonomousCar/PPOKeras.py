import numpy as np
import keras
from keras import layers
from keras import backend as K
from keras import Model
from numpy import newaxis
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mlagents.envs import UnityEnvironment
import cv2


def PPO_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred # maybe need K.sum
        old_prob = y_true* old_prediction
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r* advantage, K.clip(r, min_value=1 - self.epsilon, max_value= 1 + self.epsilon)*advantage))
    return loss

class PPOAgent:
    def __init__(self, env, n_actions, n_features, action_low=-1, action_high=1, reward_decay=0.99,
                 actor_learning_rate=0.01, critic_learning_rate=0.01, learning_rate_decay=0.95,
                 ):
        self.env = env
        self.state_size = n_features
        self.action_size = n_actions
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = reward_decay   # discount rate
        self.vector_observation_shape = [None, self.state_size]
        self.visual_observation_shape = [None, 16, 40, 1]
        self.action_shape = [None, self.action_size]
        self.actor_model = self.actor_network()
        self.critic_model= self.critic_network()
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate # often larger than actor_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = 64
        self.epsilon = 0.2 # used to clip
        self.entfact = 2e-2 # entropy factor, to encourage exploration
        self.lam = 0.95 # gae factor
        self.memory = [] # store (s, a, r) for one agent
        self.agents = 5 # number of agents that collect memory
        self.history = {} # store the memory for different agents
        self.history['states'] = []
        self.history['observations'] = []
        self.history['actions'] = []
        self.history['discounted_rs'] = []
        self.history['advantages'] = []

    


    def actor_network(self):
        in1 = layers.Input(shape=self.visual_observation_shape, name="visual_state")
        in2 = layers.Input(shape=self.vector_observation_shape, name="vector_state")
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_size,)
        l1 = layers.concatenate([in1, in2])
        l2 = layers.Dense(320, activation='relu')(l1)
        l3 = layers.Dense(320, activation='relu')(l2)
        l4 = layers.Dense(320, activation='relu')(l3)
        action = layers.Dense(self.action_shape, activation="tanh")(l4) # need to change teh output activation function
        actor = Model(inputs=[in1,in2, advantage, old_prediction], outputs=action)
        actor.compile(optimizer=keras.optimizers.Adam(lr=self.actor_learning_rate),
                                                      loss=[PPO_loss(
                                                          advantage=advantage,
                                                          old_prediction=old_prediction)])
        return actor
       
    def critic_network(self):
        in1 = layers.Input(shape=self.visual_observation_shape, name="visual_state")
        in2 = layers.Input(shape=self.vector_observation_shape, name="vector_state")
        in3 = layers.Input(shape=self.action_shape, name="action")
        l1 = layers.concatenate([in1, in2, in3])
        l2 = layers.Dense(320, activation='relu')(l1)
        l3 = layers.Dense(320, activation="relu")(l2)
        l4 = layers.Dense(320, activation='relu')(l3)
        self.value = layers.Dense(1)(l4)
        critic = Model(inputs=[in1, in2, in3], outputs=value)
        critic.compile(optimizer=keras.optimizers.Adam(lr=self.critic_learning_rate), loss='mse')
        return critic

    def choose_action(self, vector_observation, visual_observation, train=True):
        #maybe need to reshape vector_observation
        # need different action for train and test
        a = self.actor_model.predict(vector_observation, visual_observation)
        return np.clip(a, self.action_low, self.action_high)

    def remember(self, vector_observation, visual_observation, action, reward, next_vector_observation, next_visual_observation):
        self.memory += [[vector_observation[0], visual_observation[0], action, reward, next_vector_observation[0], next_visual_observation[0]]]

    def discount_rewards(self, rewards, gamma, value_next=0.0):
        discounted_r = np.zeros_like(rewards)
        running_add = value_next
        for t in reversed(range(0, len(rewards))):
            discounted_r[t] = running_add = running_add * gamma + rewards[t]
        return discounted_r
    
    def process_memory(self):
        memory = np.vstack(np.array(self.memory))
        vector_states = np.vstack(memory[:,0])
        visual_observations = np.vstack(memory[:,1]).reshape((-1, 16,40,1))
        actions = np.vstack(memory[:,2])
        rewards = memory[:,3]
        last_next_vector_state = memory[:,4][-1]
        last_next_visual_observation = memory[:,5][-1]
        discounted_episode_rewards = self.discount_rewards(rewards, self.gamma)[:,newaxis]
        value_estimates = self.critic_model(vector_states, visual_observations) # maybe need to flatten
        value_estimates = np.append(value_estimates, 0) # no clue why
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

        actor_loss = self.actor_model.fit([vector_observations,visual_observations,actiadvantageons],, batch_size=self.batch_size, shuffle=True, epochs=10, verbose=False)
        critic_loss = self.critic_model.fit(, batch_size=self.batch_size, shuffle=True, epochs=10, verbose=False)


        for key in self.history:
            self.history[key]
        
#-------------------------------------------------------------------------------------
env_name = "RL-AutonomousCar"
#file_name=env_name
env = UnityEnvironment(file_name=env_name)

# Examine environment parameters
print(str(env))

# Set the default brain to work with
default_brain = env.brain_names[1]
brain = env.brains[default_brain]

def edges(img, thr1=100, thr2=150):
    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(np.uint8(imgBlur*255),thr1,thr2)/255
    return edges.reshape(-1,*edges.shape,1)

def showEdges(img, thr1=100, thr2=150):
    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(np.uint8(imgBlur*255),thr1,thr2)/255
    # gr = img[...,:3].dot([0.299, 0.587, 0.114])
    plt.subplots(figsize=(15,15))
    plt.subplot(131), plt.xticks([]), plt.yticks([]), plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(132), plt.xticks([]), plt.yticks([]), plt.title('Blurred Image')
    plt.imshow(imgBlur)
    plt.subplot(133), plt.xticks([]), plt.yticks([]), plt.title('Edges')
    plt.imshow(edges, cmap='gray')
    plt.show()

# Reset the environment
#env_info = env.reset(train_mode=False)[default_brain]
env_info = env.reset()
brainInfo = env_info['CarLearning']
print(np.array(brainInfo.visual_observations).shape)
#print(env_info.states)

for observation in brainInfo.visual_observations:
    print("Agent observations look like:")
    if observation.shape[3] == 3:
        plt.imshow(observation[0,:,:,:])
    else:
        plt.imshow(observation[0,:,:,0])

img = brainInfo.visual_observations[0][0]
#plt.imshow(img)
#plt.show()
showEdges(img)

for i in range(40):
    for _ in range(10):
        env_info = env.step([1,0])[default_brain]
        #showEdges(env_info.visual_observations[0][0])
#     print(str((i+1)*10)+' steps')
    img = env_info.visual_observations[0][0]
    if i==0:
        sky = img[:3,:]
    else:
        sky += img[:3,:]
    #showEdges(img)
sky /= 40
print('State is :', env_info.vector_observations[0])

plt.imshow(sky)
plt.show()

agent = PPOAgent(env,
                n_actions=2,
                n_features=1,
                actor_learning_rate=1e-5,
                critic_learning_rate=2e-5
                )
rewards = []

# PPO
n_episodes = 200

# to-do load model
for i_episode in range(n_episodes):
    env_info = agent.env.reset(train_mode=True)[default_brain]
    state = env_info.vector_observations
    env_info.visual_observations[0][0][:3,:] = sky # pretend that we always see the sky above
    observation = edges(env_info.visual_observations[0][0])
    r = 0
    while True:
        action = agent.choose_action(state, observation)
        env_info = agent.env.step(action)[default_brain]
        next_state = env_info.vector_observations[0]
        env_info.visual_observations[0][0][:3,:] = sky # pretend that we always see the sky above
        next_observation = edges(env_info.visual_observations[0][0])
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        r += reward
        agent.remember(state, observation, action, reward, next_state, next_observation)
        state = next_state
        observation = next_observation
        if done:
            print("episode:", i_episode+1, "rewards: %.2f" % r, end="\r")
            agent.process_memory()
            rewards += [r]
            break
    if (i_episode+1) % agent.agents == 0: # update every n_agent episodes
        agent.replay()
    if (i_episode+1) % 100 == 0:
         agent.saver.save(agent.sess, "model/keras_model_PPO.ckpt");
agent.saver.save(agent.sess, "model/keras_model_PPO.ckpt");
print("\n")
print("finished learning!")