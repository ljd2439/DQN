from dqn import DQN
from collections import deque
import tensorflow as tf
import numpy as np
import random
import copy

GAMMA = 0.95
EPSILON_DECAY = 0.999
LEARNING_RATE = 0.005

	

class Agent:
	def __init__(self, state_dim, action_dim):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.memory = deque(maxlen=2000)
		self.epsilon = 1.0            # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = EPSILON_DECAY
		self.learning_rate = LEARNING_RATE
		self.gamma = GAMMA    # discount rate

		self.network = DQN(self.state_dim, self.action_dim, self.learning_rate)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state, is_expl=True):
		# based on epsilon greedy policy
		if is_expl and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_dim) # exploration
		act_value = self.network.predict(state)[0]               # exploitation
		return np.argmax(act_value)  # return action index

	def network_update(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		loss = 0
		for state, action, reward, next_state, done in minibatch:
			y_val = self.network.predict(state)[0]
			y_val[action] = reward
			if not done:
				y_val[action] = (reward + self.gamma * np.amax(self.network.predict(state)[0]))
			loss += self.network.fit(y_val, state)
		#print "loss : ", loss/batch_size
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.network.load_weights(name)

	def save(self, name):
		self.network.save_weights(name)


