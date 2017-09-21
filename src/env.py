from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import random
import copy

RED = 0
GREEN = 1

class Environment:
	def __init__(self, state_dim, action_dim):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.state = np.ndarray(shape=(self.state_dim[0], self.state_dim[1], 3))

		self.green_num = GREEN
		self.remain_green_num = GREEN
		self.red_num = RED

	def reset(self):
		self.state.fill(0)

		'''
		self.remain_green_num=GREEN
		self.state[self.state_dim[0]-1][self.state_dim[1]-1][1] = 1
		self.state[0][0][2] = 1
		'''
		random_list = range(0, self.state_dim[0]*self.state_dim[1])
		random.shuffle(random_list)
		# ====== state setting ======
		# set red = 0, green = 1, agent = 2
		# state[][][0] = red state
		# state[][][1] = green state
		# state[][][2] = agent state
		# =====================
		# random list [0 ~ red_num] : red
		# random list [red_num ~ (red_num + green_num)] : green
		# random list [(red_num + green_num) ~ (red_num + green_num+1)] : agent
		# =====================

		for i in range(self.red_num):
			red_x = random_list[i]/self.state_dim[1]
			red_y = random_list[i]%self.state_dim[1]
			self.state[red_x][red_y][0] = 1

		for i in range(self.red_num, self.red_num+self.green_num):
			green_x = random_list[i]/self.state_dim[1]
			green_y = random_list[i]%self.state_dim[1]
			self.state[green_x][green_y][1] = 1
		self.remain_green_num = GREEN

		i = self.green_num+self.red_num
		agent_x = random_list[i]/self.state_dim[1]
		agent_y = random_list[i]%self.state_dim[1]
		self.state[agent_x][agent_y][2] = 1

	def getState(self, is_copy=True):
		if is_copy:
			return copy.deepcopy(self.state)
		else:
			return self.state

	def step(self, action):
		agent_pos = [0,0]
		cur_agent_x = 0
		cur_agent_y = 0
		for i in range(self.state_dim[0]):
			for j in range(self.state_dim[1]):
				if self.state[i][j][2] == 1:
					cur_agent_x = i
					cur_agent_y = j
					agent_pos = [cur_agent_x, cur_agent_y]

		# position example
		# [0,0] [0,1] [0,2] [0,3]
		# [1,0] [1,1] [1,2] [1,3]
		# [2,0] [2,1] [2,2] [2,3]
		# [3,0] [3,1] [3,2] [3,3]

		reward = 0
		if action == 0: # up
			if agent_pos[0] > 0:
				agent_pos[0] -= 1
		elif action == 1: #down
			if agent_pos[0] < self.state_dim[0]-1:
				agent_pos[0] += 1
		elif action == 2: #left
			if agent_pos[1] > 0:
				agent_pos[1] -= 1
		elif action == 3: #right
			if agent_pos[1] < self.state_dim[1]-1:
				agent_pos[1] += 1

		if self.state[agent_pos[0]][agent_pos[1]][0] == 1:
			self.state[agent_pos[0]][agent_pos[1]][0] = 0
			reward = -1
		if self.state[agent_pos[0]][agent_pos[1]][1] == 1:
			self.state[agent_pos[0]][agent_pos[1]][1]  = 0
			self.remain_green_num -= 1
			reward = 1

		self.state[cur_agent_x][cur_agent_y][2] = 0
		self.state[agent_pos[0]][agent_pos[1]][2] = 1

		done = False
		if self.remain_green_num == 0:
			done = True

		return copy.deepcopy(self.state), reward, done

	def render(self):
		grid_len = 30
		left_top_x = -((self.state_dim[1]+1)/2) * grid_len
		left_top_y = ((self.state_dim[0]+1)/2) * grid_len
		right_bottom_x = ((self.state_dim[1])/2) * grid_len
		right_bottom_y = -((self.state_dim[0])/2) * grid_len

		#grid rendering
		glColor3f(0.0, 0.0, 0.0)
		for i in range(self.state_dim[0]+1):
			glBegin(GL_LINES)
			glVertex3f(left_top_x, left_top_y-i*grid_len, 0)
			glVertex3f(right_bottom_x, left_top_y-i*grid_len, 0)
			glEnd()

		for i in range(self.state_dim[1]+1):
			glBegin(GL_LINES)
			glVertex3f(left_top_x+i*grid_len, left_top_y, 0)
			glVertex3f(left_top_x+i*grid_len, right_bottom_y, 0)
			glEnd()

		#red, green, agent rendering

		for i in range(self.state_dim[0]):
			for j in range(self.state_dim[1]):
				if self.state[i][j][0] == 1:
					glColor3f(1.0, 0.0, 0.0)
					glPushMatrix()
					glTranslatef(left_top_x + j*grid_len + grid_len/2,
						left_top_y - i*grid_len - grid_len/2, 0)
					glutSolidCube(grid_len)
					glPopMatrix()

				if self.state[i][j][1] == 1:
					glColor3f(0.0, 1.0, 0.0)
					glPushMatrix()
					glTranslatef(left_top_x + j*grid_len + grid_len/2,
						left_top_y - i*grid_len - grid_len/2, 0)
					glutSolidCube(grid_len)
					glPopMatrix()

				if self.state[i][j][2] == 1:
					glColor3f(0.0, 0.0, 0.0)
					glPushMatrix()
					glTranslatef(left_top_x + j*grid_len + grid_len/2,
						left_top_y - i*grid_len - grid_len/2, 0)
					glutSolidCube(grid_len)
					glPopMatrix()


