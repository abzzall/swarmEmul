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
	Vx = V[:, :, 2, 0]
	Vy = V[:, :, 2, 1]
	Vd = numpy.power(Vx, 2) + numpy.power(Vy, 2)
	return numpy.mean(numpy.mean(Vd, axis=1))


def save_episode(env: Env, directory, run_number):
	env.save_episode(os.path.join(directory, str(run_number)))


def show_episode(directory, run_number, window_width=WINDOW_SIZE, window_height=WINDOW_SIZE, fps=FPS):
	episode_replay_from_file(os.path.join(directory, str(run_number) + '.npz'), window_width, window_height, fps)


def fill_report_with_graph(directory, input, writer, outs=[2, 3, 4, 5, 6, 7], ax_rows=3, ax_cols=2, row_number=101):
	df = pandas.read_excel(os.path.join(directory, 'report.xls'))
	df.set_index('id', inplace=True)
	print(df.columns)
	fig, ax = plt.subplots(nrows=ax_rows, ncols=ax_cols, figsize=(15, 10))
	k = 0
	for i in range(ax_rows):
		for j in range(ax_cols):
			if k == len(outs):
				break
			df.plot(ax=ax[i][j], x=df.columns[input], y=df.columns[outs[k]], marker='o', grid=True)
			k = k + 1

	df.to_excel(writer, sheet_name=directory)
	workbook = writer.book
	sheet = writer.sheets[directory]

	r = 1
	for out in outs:
		chart = workbook.add_chart({'type': 'line'})
		chart.add_series(
			{
				'categories': [directory, 1, input + 1, 1 + row_number, input + 1],
				'values': [directory, 1, out + 1, 1 + row_number, out + 1],
				'marker': {'type': 'automatic', 'fill': {'color': 'red'}},
				'name': df.columns[out]
			}
			)
		sheet.insert_chart('K' + str(r), chart, {'x_scale': 3, 'y_scale': 1})
		r += 20
