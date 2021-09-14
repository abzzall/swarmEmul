from env import Env
from geometry import getDistance
import numpy
import os
import pandas
import numpy
import matplotlib
from constants import *
from matplotlib import pyplot as plt
from env import *
from visualiser import *


def alive_agent_count(env: Env):
	s = 0
	for agent in env.agents:
		if not agent.is_dead:
			s += 1
	return s


def leader_goal_distance(env: Env):
	return getDistance(env.xL, env.yL, env.xG, env.yG)


def meanV(V):
	Vx = V[:, :, 3, 0]
	Vy = V[:, :, 3, 1]
	Vd = numpy.power(Vx, 2) + numpy.power(Vy, 2)
	return numpy.mean(numpy.mean(Vd, axis=1))

def save_episode(env:Env, directory, run_number):
	env.save_episode( os.path.join(directory, str(run_number)))

def show_episode(directory, run_number, window_width=WINDOW_SIZE, window_height=WINDOW_SIZE, fps=FPS):
	episode_replay_from_file(os.path.join(directory, str(run_number)+'.npz'),window_width, window_height, fps)
if __name__ == '__main__':
	directory = 'run'
	if not os.path.exists(directory):
		os.makedirs(directory)

	SENSOR_RANGE = 5

	ROBOT_NUMBER = 9
	DX = [-SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0,
	      SENSOR_RANGE + ROBOT_RADIUS, -ROBOT_RADIUS - SENSOR_RANGE, 0, ROBOT_RADIUS + SENSOR_RANGE]
	DY = [-SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, 0, 0,
	      SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS]

	env = Env(
		width=ENV_SIZE, height=ENV_SIZE, goal_x=GOAL_X, goal_y=GOAL_Y, N=ROBOT_NUMBER,
		obstacle_pos=OBSTACLE_POS,
		desired_X=DX, desired_Y=DY, sensor_range=SENSOR_RANGE, leader_x=XL, leader_y=YL, robot_radius=ROBOT_RADIUS,
		sensor_detection_count=SENSOR_DETECTION_COUNT, buffer_size=MAX_T
	)

	df = pandas.DataFrame(
		columns=['id', 'Чувствительность лидара', 'Вес скорости избегания столкновении',
		         'Вес скорости сохранения формы', 'Вес скорости достижения цели', 'Количество выживших агентов',
		         'Расстояние между виртуальным лидером и целевой точки',
		         'Количество итерации', 'Средний скорость']
	)
	# df.set_index('id', inplace=True)

	w2 = 1
	w3 = 1
	run = 0
	for w1 in numpy.arange(0, 10, 0.1):
		print(w1)
		print(w2)
		print(w3)
		episode_gui(env, w1, w2, w3)

		df.loc[run] = [run, SENSOR_RANGE, w1, w2, w3, alive_agent_count(env), leader_goal_distance(env), env.t,
		               meanV(env.v_history)]
		print(env.t)
		env.save_episode(os.path.join(directory, str(run)))
		run = run + 1
def fill_report_with_graph(directory, input, outs=[4, 5, 6, 7], full_report=full_report, ax_rows=2, ax_cols=2, row_number=101):
	df=pandas.read_excel(os.path.join(directory, 'report.xls'))
	df.set_index('id', inplace=True)
	print(df.columns)
	fig, ax=plt.subplots(nrows=ax_rows, ncols=ax_cols, figsize=(15, 6))
	k=0
	for i in range(ax_rows):
		for j in range(ax_cols):
			if k==len(outs):
				break
			df.plot(ax=ax[i][j] ,x=df.columns[input] , y=df.columns[outs[k]], marker='o', grid=True)
			k=k+1
	writer = pandas.ExcelWriter(full_report, engine='xlsxwriter')
	df.to_excel(writer, sheet_name=directory)
	workbook = writer.book
	sheet=writer.sheets[directory]

	r=1
	for out in outs:
		chart = workbook.add_chart({'type': 'line'})
		chart.add_series({'values': [directory, 1, input, 1+row_number, input],
						 'categories': [directory, 1,out, 1+row_number,  out],
						 'marker': {'type': 'automatic'},
						 'name': df.columns[out]})
		sheet.insert_chart('K'+str(r), chart, {'x_offset': 25, 'y_offset': 10})
		r+=20
	workbook.close()
	writer.close()