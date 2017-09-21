import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('../base')

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np

from env import Environment
from agent import Agent

WIN_WIDTH = 960
WIN_HEIGHT = 960

STATE_DIM = [3,3]
ACTION_DIM = 4
BATCH_SIZE = 32

MAX_STEP = 100
EPISODE = 4000

USE_RECENT_CKPT = False

TRAINING = False
PLAYING = False

ACTION_NAME = ['Up', 'Down', 'Left', 'Right']

# name of weight data
# you can load and save by this name
saved_weight = "../data/saved_weight_12"

class Experiment:
	def __init__(self):
		self.state_dim = STATE_DIM	# m x n   grid
		self.action_dim = ACTION_DIM 	# action : up, down, right, left
		self.env = Environment(self.state_dim, self.action_dim)
		self.agent = Agent(self.state_dim, self.action_dim)

		self.episode = 0
		self.batch_size = BATCH_SIZE

		self.isTraining = TRAINING
		self.isPlaying = PLAYING

		#======opengl setting======
		argv = sys.argv
		glutInit(argv)
		glutInitWindowPosition(0,0)
		glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT)
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
		glutCreateWindow("DQN example")
		glutDisplayFunc(self.display)
		glutReshapeFunc(self.reshape)
		glutKeyboardFunc(self.keyCB)
		#======================

		if USE_RECENT_CKPT:
			print "main.59.Load Recent CKPT"
			self.agent.load(saved_weight)
		else:
			print "main.62.Use New CKPT"

		self.env.reset()
		self.step=0
		self.timer_func()
		glutMainLoop()

	def timer_func(self, fps=200):
		if self.isPlaying:
			state = self.env.getState()
			action = self.agent.act(state, is_expl=False)
			next_state, reward, done = self.env.step(np.argmax(action))
			if done:
				self.isPlaying = False

		if self.isTraining:
			state = self.env.getState()
			action = np.argmax(self.agent.act(state))
			next_state, reward, done = self.env.step(action)

			self.agent.remember(state, action, reward, next_state, done)
			self.step += 1

			if self.step % 30 == 29:
				if len(self.agent.memory) > self.batch_size:
					self.agent.network_update(self.batch_size)

			if done or self.step == MAX_STEP:
				if len(self.agent.memory) > self.batch_size:
					self.agent.network_update(self.batch_size)

				print("episode: {}/{}, exploration epsilon: {:.2}\n".format(self.episode, EPISODE, self.agent.epsilon))

				self.episode += 1
				if self.episode >= EPISODE:
					print "main.71.Network Saved!"
					self.agent.save(saved_weight)
					self.isTraining = False

				self.step = 0
				self.env.reset()

		glutPostRedisplay()
		glutTimerFunc(int(1000/fps), self.timer_func, fps)

	def keyCB(self, key, x, y):
		if key == 'q': # quit
			glutDestroyWindow (1)
		if key == 'r':
			#reset
			self.step=0
			self.env.reset()
			self.isPlaying=False
			self.isTraining=False
		if key == 't':
			if not self.isTraining:
				self.env.reset()
				self.isTraining=True
				self.isPlaying=False
			elif self.isTraining:
				print "main.116.save network : ", saved_weight
				self.agent.save(saved_weight)
				self.isTraining=False
		if key == ' ': # in playing mode, you can replay by push the 'p' button
			self.isPlaying= not self.isPlaying
			self.isTraining=False
		if key == 'n':
			state = self.env.getState()
			action = self.agent.act(state, is_expl=False)
			print "main.130.",
			print "Action : ", ACTION_NAME[np.argmax(action)]
			print action

		#if key == 's': # in training mode, you can save by push the 's' button
			#self.agent.save(saved_weight)

		if key=='w':
			self.env.step(0)
			state = self.env.getState()
			action = np.argmax(self.agent.act(state))
		if key=='s':
			self.env.step(1)
		if key=='a':
			self.env.step(2)
		if key=='d':
			self.env.step(3)

	def display(self):
		glClearColor(0.7, 0.7, 0.7, 0.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		gluLookAt(0, 0, 30, 0, 0, 0, 0, 1, 0)

		glPushMatrix()
		self.env.render()
		glPopMatrix()

		glutSwapBuffers()

	def reshape(self, w, h):
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-WIN_WIDTH/2, WIN_WIDTH/2, -WIN_HEIGHT/2, WIN_HEIGHT/2, -30, 30)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

if __name__=="__main__":
	Experiment()
