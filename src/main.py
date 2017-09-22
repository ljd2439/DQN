from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import sys

from env import Environment
from agent import Agent

WIN_WIDTH = 960
WIN_HEIGHT = 960

STATE_DIM = [4,4]
ACTION_DIM = 4
BATCH_SIZE = 32

MAX_STEP = 100
EPISODE = 1000

TRAINING = True
PLAYING = False

# name of weight data
# you can load and save by this name
saved_weight = "../data/saved_weight_5"

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

		if self.isTraining:
			self.training()
		else:
			if self.isPlaying:
				self.agent.load(saved_weight)
				self.playing()

		glutMainLoop()

	def playing(self):
		self.env.reset()
		self.timer_func()
		print "new play"

	def training(self):
		if self.episode < EPISODE+1:
			self.step = 0
			self.env.reset()
			self.timer_func()
		else:
			self.agent.save(saved_weight)

	def timer_func(self, fps=200):
		state = self.env.getState()
		action = self.agent.act(state)
		next_state, reward, done = self.env.step(action)
		glutPostRedisplay()

		if self.isPlaying:
			if not done:
				glutTimerFunc(int(1000/fps), self.timer_func, fps)

		if self.isTraining:
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
				self.training()
			else:
				glutTimerFunc(int(1000/fps), self.timer_func, fps)

	def keyCB(self, key, x, y):
		if key == 'q': # quit
			glutDestroyWindow (1)

		if key == 'p': # in playing mode, you can replay by push the 'p' button
			self.playing()

		if key == 's': # in training mode, you can save by push the 's' button
			self.agent.save(saved_weight)

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
