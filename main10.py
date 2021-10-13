from random import randint
from random import seed
from typing import List
import numpy

from env import Env
from visualiser import episode_gui
from datetime import datetime


def generate_dxy(N: int, distance:int=5):
	X=numpy.array([(i-(N//2))*distance for i in range(N)]*N)
	Y=numpy.zeros(N*N)
	for i in range(N):
		for j in range(N):
			Y[i*N+j]=(i-(N//2))*distance
	return X, Y
TMax=10_000

N=10
def random_xs():
	seed(0)

	s=0
	xs=[]
	for i in range(9):
		s+=randint(3, 10)
		xs.append(s)
	return xs

def main():
	#TMax = 10_000

	#N = 100
	WINDOW_SIZE=1000
	goal_x=900
	goal_y=900
	Dx, Dy=generate_dxy(N)
	env = Env(
		width=WINDOW_SIZE, height=WINDOW_SIZE, goal_x=goal_x, goal_y=goal_y, N=N*N,
		desired_X=Dx, desired_Y=Dy, leader_x=100, leader_y=100, robot_radius=1,
		buffer_size=TMax
	)
	# env.episode(1, 1)
	# print(env.t)
	episode_gui(env, 1, 1, paused=True)
	print(env.alive_agent_count())


if __name__=='__main__':
	print(datetime.now().strftime("%H:%M:%S"))
	main()
	print(datetime.now().strftime("%H:%M:%S"))
