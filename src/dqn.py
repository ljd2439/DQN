import tensorflow as tf
import numpy as np

class DQN:
	def __init__(self, state_dim, action_dim, learning_rate):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate

		self.set_network()

	def set_network(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.input_layer = tf.placeholder(tf.float32,
				[None, self.state_dim[0], self.state_dim[1], 3], name='input')

			conv1 = tf.layers.conv2d(
				inputs=self.input_layer,
				filters=16,
				kernel_size=[5,5],
				padding="same",
				activation=tf.nn.relu,
				name = 'conv1')

			#pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
			pool1 = conv1

			conv2 = tf.layers.conv2d(
				inputs=pool1,
				filters=32,
				kernel_size=[5,5],
				padding="same",
				activation=tf.nn.relu,
				name = 'conv2')

			#pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
			pool2 = conv2

			# You have to calculate the flatten size
			# in this case, size is state_dim[0] * state_dim[1] * filter_size.
			# because there is no pooling.
			pool2_flat = tf.reshape(pool2, [-1, self.state_dim[0]*self.state_dim[1]*32])

			dense1 = tf.layers.dense(inputs=pool2_flat, units=128, name = 'dense1')
			dense2 = tf.layers.dense(inputs=dense1, units=4, name = 'dense2')

			self.predict_val = tf.nn.softmax(dense2, name="prob")
			self.y_val = tf.placeholder(tf.float32, [None, self.action_dim])
			self.loss = tf.reduce_mean(tf.square(self.y_val - self.predict_val))
			self.opt = tf.train.GradientDescentOptimizer(self.learning_rate, name="opt_GD")
			self.train = self.opt.minimize(self.loss)

			self.saver = tf.train.Saver()
			self.sess = tf.Session(graph=self.graph)
			self.sess.run(tf.global_variables_initializer())

	def predict(self, state):
		return self.sess.run(self.predict_val,
			feed_dict={
				self.input_layer : np.reshape(state, (-1, self.state_dim[0], self.state_dim[1], 3))
				})

	def fit(self, target, state):
		train, loss = self.sess.run((self.train, self.loss), feed_dict={
				self.y_val : np.reshape(target, (-1, self.action_dim)),
				self.input_layer : np.reshape(state, (-1, self.state_dim[0], self.state_dim[1], 3)),
			} )
		return loss

	def load_weights(self, name):
		self.saver.restore(self.sess, name)

	def save_weights(self, name):
		if self.saver.save(self.sess, name):
			print name, ", saved"
