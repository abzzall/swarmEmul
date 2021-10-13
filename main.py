from random import randint

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

def perebor(a, b, k):
	# print('a, b, k:'+str(a)+', '+str(b)+', '+str(k))
	if k==0:
		return [[]]
	if a+k>b+1 or k<1:
		return []
	elif a+k==b+1:
		# print(range(a, b+1))
		return [[*range(a, b+1)]]
	elif k==1:
		# print([[i] for i in range(a, b+1)])
		return [[i] for i in range(a, b+1)]
	else:
		result=[]
		for i in range(a, b-k+1):
			try:
				p=perebor(i+1, b, k-1)
				result=result+[[i]+l for l in p]
			except Exception:
				# p=perebor(i+1, b, k-1)

				# print('a, b, k, i:' + str(a) + ', ' + str(b) + ', ' + str(k)+ ', ' + str(i))
				raise Exception('a, b, k, i, p:' + str(a) + ', ' + str(b) + ', ' + str(k)+ ', ' + str(i))
		return result+[[*range(b-k+1, b+1)]]
def random_xs():
	seed(0)

	s=0
	xs=[]
	for i in range(9):
		s+=randint(3, 10)
		xs.append(s)
	return xs
def kill_list():
	p=[perebor(0, 8, i) for i in range(9)]
	s=1
	i=0
	killed_list=[]
	while s>0:
		s=0
		for l in p:
			if len(l)>i:
				killed_list=killed_list+[l[i]]
				s+=1
		i+=1
	return killed_list

def main():
	p=[perebor(0, 8, i) for i in range(9)]
	s=1
	i=0
	killed_list=[]
	while s>0:
		s=0
		for l in p:
			if len(l)>i:
				killed_list=killed_list+[l[i]]
				s+=1
		i+=1
	# print(len(killed_list))


	# print(perebor(0, 8, 3))
	# p=perebor(0, 8, 3)
	seed(1)

	s=0
	xs=[]
	for i in range(9):
		s+=randint(3, 10)
		xs.append(s)
	env = Env(
		width=100, height=100, goal_x=50, goal_y=80, N=ROBOT_NUMBER,
		desired_X=DX, desired_Y=DY, leader_x=50, leader_y=20, robot_radius=ROBOT_RADIUS,
		buffer_size=MAX_T
	)
	parent_directory=os.path.join('..', 'run1')
	if not os.path.exists(parent_directory):
		os.makedirs(parent_directory)

	for g, killed_agents in enumerate(killed_list):
		# for goal_y in range(80, 0, -20):
			directory = os.path.join(parent_directory, str(g))
			if not os.path.exists(directory):
				os.makedirs(directory)

			img_directory = os.path.join(directory, 'plt')
			if not os.path.exists(img_directory):
				os.makedirs(img_directory)

			# env.yG=goal_y
			df = pandas.DataFrame(
				columns=['id', 'The weight of the speed of keeping the formation', 'The weight of the speed of moving to the goal',
				         'Количество выживших агентов',
				         'Расстояние между виртуальным лидером и целевой точки', 'Количество итерации движение строя',
				         'Средний скорость', 'The step\'s count of forming desired pattern',
				         'The step\'s count of reaching the goal']
			)
			df.set_index('id', inplace=True)
			# df1 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
			# df2 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
			# df3 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
			# df4 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
			# df5 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
			# df6 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])

			np1 = numpy.zeros((100, 100))
			np2 = numpy.zeros((100, 100))
			np3 = numpy.zeros((100, 100))
			np4 = numpy.zeros((100, 100))
			np5 = numpy.zeros((100, 100))
			np6 = numpy.zeros((100, 100))
			run = 0

			for n1 in range(1, 101):
				# df1_row = [0] * 100
				# df2_row = [0] * 100
				# df3_row = [0] * 100
				# df4_row = [0] * 100
				# df5_row = [0] * 100
				# df6_row = [0] * 100

				for n2 in range(1, 101):
					w1 = n1 / 10
					w2 = n2 / 10
					print(g)
					print(killed_agents)
					# print(goal_y)
					print(run)

					# env.episode(w1, w2, reset_to_line=True, killed_agents=killed_agents)
					# 		episode_gui(env, w1, w2)
					# 		episode_gui(env, w1, w2)
					env.reset()
					env.reset_to_custom_line_formation(xs, 20)
					# env.reset_to_line_formation()

					for i in killed_agents:
						env.agents[i].is_dead=True
					env.reset_leader()
					while not env.is_done:
						env.play_step(w1, w2)
					aac=alive_agent_count(env)
					lgd=leader_goal_distance(env)
					row = [w1, w2, alive_agent_count(env), leader_goal_distance(env), env.t, meanV(env.v_history),
					       env.tForm, env.tGoal]
					df.loc[run] = row
					print(row)
					# if aac<9-len(killed_agents) or not equals(lgd, 0.0):
					# 	raise Exception(str(row)+str(killed_agents))
					env.save_episode(os.path.join(directory, str(run)))

					np1[n1 - 1, n2 - 1] = row[2]
					np2[n1 - 1, n2 - 1]  = row[3]
					np3[n1 - 1, n2 - 1]  = row[4]
					np4[n1 - 1, n2 - 1]  = row[5]
					np5[n1 - 1, n2 - 1]  = row[6]
					np6[n1 - 1, n2 - 1]  = row[7]

					run = run + 1

			writer = pandas.ExcelWriter(os.path.join(directory, 'report.xls'), engine='xlsxwriter')

			df.to_excel(writer, sheet_name='report')
			writer.close()
			W1 = numpy.outer(numpy.array(range(1, 101)) / 10, numpy.ones(100))
			W2 = W1.copy().T

			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			ax.azim = 60
			ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[6])
			ax.plot_surface(W1, W2, np5, cmap='viridis', edgecolor='green')
			fig.savefig(os.path.join(img_directory, df.columns[6] + '.png'))

			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			ax.azim = 60
			ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[7])
			ax.plot_surface(W1, W2, np6, cmap='viridis', edgecolor='green')
			fig.savefig(os.path.join(img_directory, df.columns[7] + '.png'))

			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			ax.azim = 60
			ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[4])
			ax.plot_surface(W1, W2, np3, cmap='viridis', edgecolor='green')
			fig.savefig(os.path.join(img_directory, df.columns[4] + '.png'))

			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			ax.azim = 60
			ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[5])
			ax.plot_surface(W1, W2, np4, cmap='viridis', edgecolor='green')
			fig.savefig(os.path.join(img_directory, df.columns[5] + '.png'))

			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			ax.azim = 60
			ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[2])
			ax.plot_surface(W1, W2, np1, cmap='viridis', edgecolor='green')
			fig.savefig(os.path.join(img_directory, df.columns[2] + '.png'))

			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			ax.azim = 60
			ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[3])
			ax.plot_surface(W1, W2, np2, cmap='viridis', edgecolor='green')
			# plt.show()
			fig.savefig(os.path.join(img_directory, df.columns[3] + '.png'))
def main1():
	seed(1)

	s=0
	xs=[]
	for i in range(9):
		s+=randint(3, 10)
		xs.append(s)
	env = Env(
		width=100, height=100, goal_x=50, goal_y=80, N=ROBOT_NUMBER,
		desired_X=DX, desired_Y=DY, leader_x=50, leader_y=20, robot_radius=ROBOT_RADIUS,
		buffer_size=MAX_T
	)
	p=kill_list()
	directory = os.path.join('..', 'run1')
	if not os.path.exists(directory):
		os.makedirs(directory)

	img_directory = os.path.join(directory, 'plt')
	if not os.path.exists(img_directory):
		os.makedirs(img_directory)

	# env.yG=goal_y
	df = pandas.DataFrame(
		columns=['id', 'The weight of the speed of maintaining the formation', 'Вес скорости достижения цели',
		         'Количество выживших агентов',
		         'Расстояние между виртуальным лидером и целевой точки', 'Количество итерации движение строя',
		         'Средний скорость', 'The count of steps to form desired pattern',
		         'Количество итерации достижения цели', 'Количество crashed агентов']
	)
	df.set_index('id', inplace=True)
	# df1 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df2 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df3 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df4 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df5 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df6 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])

	np1 = numpy.zeros((100, 9))
	np2 = numpy.zeros((100, 9))
	np3 = numpy.zeros((100, 9))
	np4 = numpy.zeros((100, 9))
	np5 = numpy.zeros((100, 9))
	np6 = numpy.zeros((100, 9))
	run = 0

	for n1 in range(1, 101):
		# df1_row = [0] * 100
		# df2_row = [0] * 100
		# df3_row = [0] * 100
		# df4_row = [0] * 100
		# df5_row = [0] * 100
		# df6_row = [0] * 100

		for n2, k in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8]):             # [0, 9, 18, 27, 36, 45, 54, 63, 72]):   #
			killed_agents=p[k]
			w1 = n1 / 10
			w2 = 3
			# print(g)
			print(killed_agents)
			# print(goal_y)
			print(run)

			# env.episode(w1, w2, reset_to_line=True, killed_agents=killed_agents)
			# 		episode_gui(env, w1, w2)
			# 		episode_gui(env, w1, w2)
			env.reset()
			env.reset_to_custom_line_formation(xs, 20)
			# env.reset_to_line_formation()
			# env.reset_to_unsimmetric_line_formation()
			for i in killed_agents:
				env.agents[i].is_dead = True
			env.reset_leader()
			while not env.is_done:
				env.play_step(w1, w2)
			aac = alive_agent_count(env)
			lgd = leader_goal_distance(env)
			row = [w1, w2, alive_agent_count(env), leader_goal_distance(env), env.t, meanV(env.v_history),
			       env.tForm, env.tGoal, n2]
			df.loc[run] = row
			print(row)
			# if aac < 9 - len(killed_agents) or not equals(lgd, 0.0):
			# 	raise Exception(str(row) + str(killed_agents))
			env.save_episode(os.path.join(directory, str(run)))

			np1[n1 - 1, n2 ] = row[2]
			np2[n1 - 1, n2 ] = row[3]
			np3[n1 - 1, n2 ] = row[4]
			np4[n1 - 1, n2 ] = row[5]
			np5[n1 - 1, n2 ] = row[6]
			np6[n1 - 1, n2 ] = row[7]

			run = run + 1

	writer = pandas.ExcelWriter(os.path.join(directory, 'report.xls'), engine='xlsxwriter')

	df.to_excel(writer, sheet_name='report')
	writer.close()
	W = numpy.array(range(1, 101)) / 10
	# k=  range(9)
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[7])
	for i in range(9):
		ax.plot(W, np6[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[7] + '.png'))

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[7])
	for i in range(5):
		ax.plot(W, np6[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[7] + '3.png'))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[7])
	for i in [*range(1, 9, 2)]+[0]:
		ax.plot(W, np6[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[7] + '1.png'))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[7])
	for i in [*range(2, 9, 2)]+[0]:
		ax.plot(W, np6[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[7] + '2.png'))

	W1 = numpy.outer(numpy.array(range(1, 101)) / 10, numpy.ones(9))
	K= numpy.outer( numpy.ones(100), range(9))


	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.azim = 60
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[8], zlabel=df.columns[7])
	ax.plot_surface(W1, K, np6, cmap='viridis', edgecolor='green')
	fig.savefig(os.path.join(img_directory, df.columns[7] + '3d.png'))

#2
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[6])
	for i in range(9):
		ax.plot(W, np5[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[6] + '.png'))

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[6])
	for i in range(5):
		ax.plot(W, np5[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[6] + '3.png'))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[6])
	for i in [*range(1, 9, 2)]+[0]:
		ax.plot(W, np5[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[6] + '1.png'))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[6])
	for i in [*range(2, 9, 2)]+[0]:
		ax.plot(W, np5[:, i], label=str(i)+ ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[6] + '2.png'))

	W1 = numpy.outer(numpy.array(range(1, 101)) / 10, numpy.ones(9))
	K= numpy.outer( numpy.ones(100), range(9))


	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.azim = 60
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[8], zlabel=df.columns[6])
	ax.plot_surface(W1, K, np5, cmap='viridis', edgecolor='green')
	fig.savefig(os.path.join(img_directory, df.columns[6] + '3d.png'))

def plot_leader_goal_distance():
	env = Env(
		width=100, height=100, goal_x=50, goal_y=80, N=ROBOT_NUMBER,
		desired_X=DX, desired_Y=DY, leader_x=50, leader_y=20, robot_radius=ROBOT_RADIUS,
		buffer_size=MAX_T
	)
	killed_agents=[]
	TMAX=260
	np=numpy.zeros((100, TMAX))
	for n1 in range(1, 101):
		w1=n1/10
		w2=5
		env.reset()
		env.reset_to_custom_line_formation(random_xs(), 20)
		if killed_agents is not None:
			for i in killed_agents:
				env.agents[i].is_dead=True
		env.reset_leader()
		print(w1)
		for t in range(TMAX):
			np[n1-1, t]=leader_goal_distance(env)
			env.play_step(w1, w2)

	W = numpy.outer(numpy.array(range(1, 101)) / 10, numpy.ones(TMAX))
	T= numpy.outer(numpy.ones(100), numpy.array(range(TMAX)))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.azim = 60
	ax.set(xlabel='The weight of the speed of keeping the formation(W1)', ylabel='Step', zlabel='The distance between the leader and the goal', ylim=[10, 260])
	ax.plot_surface(W[:, 15:], T[:, 15:], np[:, 15:], cmap='viridis', edgecolor='green')

	fig.savefig('lgd.png')


#
	# fig = plt.figure(figsize=(10, 10))
	# ax = fig.add_subplot(1, 1, 1)
	# # ax.azim = 60
	# # ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[7])
	# # ax.plot_surface(W1, k, np6, cmap='viridis', edgecolor='green')
	# ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[7])
	# for i in range(9):
	# 	ax.plot(W1, np6[:, i], label=str(i)+ ' crashed')
	# ax.legend()
	# fig.savefig(os.path.join(img_directory, df.columns[7] + '.png'))
	#
	# fig = plt.figure(figsize=(10, 10))
	# ax = fig.add_subplot(1, 1, 1)
	# # ax.azim = 60
	# # ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[4])
	# # ax.plot_surface(W1, k, np3, cmap='viridis', edgecolor='green')
	# ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[4])
	# for i in range(9):
	# 	ax.plot(W1, np3[:, i], label=str(i)+ ' crashed')
	# ax.legend()
	# fig.savefig(os.path.join(img_directory, df.columns[4] + '.png'))
	#
	# fig = plt.figure(figsize=(10, 10))
	# ax = fig.add_subplot(1, 1, 1)
	# # ax.azim = 60
	# # ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[5])
	# # ax.plot_surface(W1, k, np4, cmap='viridis', edgecolor='green')
	# ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[5])
	# for i in range(9):
	# 	ax.plot(W1, np4[:, i], label=str(i)+ ' crashed')
	# ax.legend()
	# fig.savefig(os.path.join(img_directory, df.columns[5] + '.png'))
	#
	# fig = plt.figure(figsize=(10, 10))
	# ax = fig.add_subplot(1, 1, 1)
	# # ax.azim = 60
	# # ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[2])
	# # ax.plot_surface(W1, k, np1, cmap='viridis', edgecolor='green')
	# ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[2])
	# for i in range(9):
	# 	ax.plot(W1, np1[:, i], label=str(i)+ ' crashed')
	# ax.legend()
	# fig.savefig(os.path.join(img_directory, df.columns[2] + '.png'))
	#
	# fig = plt.figure(figsize=(10, 10))
	# ax = fig.add_subplot(1, 1, 1)
	# # ax.azim = 60
	# # ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[1] + '(w2)', zlabel=df.columns[3])
	# # ax.plot_surface(W1, W2, np2, cmap='viridis', edgecolor='green')
	# # plt.show()
	# ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[3])
	# for i in range(9):
	# 	ax.plot(W1, np2[:, i], label=str(i)+ ' crashed')
	# ax.legend()
	# fig.savefig(os.path.join(img_directory, df.columns[3] + '.png'))

def plot_only_formation():
	seed(1)

	s = 0
	xs = []
	for i in range(9):
		s += randint(3, 10)
		xs.append(s)
	env = Env(
		width=100, height=100, goal_x=50, goal_y=80, N=ROBOT_NUMBER,
		desired_X=DX, desired_Y=DY, leader_x=50, leader_y=20, robot_radius=ROBOT_RADIUS,
		buffer_size=MAX_T
	)
	p = kill_list()
	directory = os.path.join('..', 'run1')
	if not os.path.exists(directory):
		os.makedirs(directory)

	img_directory = os.path.join(directory, 'plt')
	if not os.path.exists(img_directory):
		os.makedirs(img_directory)

	# env.yG=goal_y
	df = pandas.DataFrame(
		columns=['id', 'The weight of the speed of maintaining the formation',
		         'Количество выживших агентов',
		         'The length of the leader\'s trajectory',
		         'Расстояние на которую сдвинулся виртуальный лидер','Угол на которую сдвинулся виртуальный лидер', 'Количество итерации движение строя',
		         'Средний скорость', 'The step\'s count of forming desired pattern',
		          'Количество crashed агентов']
	)
	df.set_index('id', inplace=True)
	# df1 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df2 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df3 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df4 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df5 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])
	# df6 = pandas.DataFrame(columns=['id'] + [i / 10 for i in range(1, 101)])

	np1 = numpy.zeros((100, 9))
	np2 = numpy.zeros((100, 9))
	np3 = numpy.zeros((100, 9))
	np4 = numpy.zeros((100, 9))
	np5 = numpy.zeros((100, 9))
	np6 = numpy.zeros((100, 9))
	run = 0
	for n2, k in enumerate( [0, 1, 2, 3, 4, 5, 6, 7, 8]):  #[0, 9, 18, 27, 36, 45, 54, 63, 72]):   #
		child_dir=os.path.join(directory, str(n2)+'_')
		if not os.path.exists(child_dir):
			os.makedirs(child_dir)
		TMax=300
		np_d=numpy.zeros((100, TMax))
		np_angle=numpy.zeros((100, TMax))
		for n1 in range(1, 101):
		# df1_row = [0] * 100
		# df2_row = [0] * 100
		# df3_row = [0] * 100
		# df4_row = [0] * 100
		# df5_row = [0] * 100
		# df6_row = [0] * 100


			killed_agents = p[k]
			w1 = n1 / 10
			# w1 = 1
			# print(g)
			print(killed_agents)
			# print(goal_y)
			print(run)

			# env.episode(w1, w2, reset_to_line=True, killed_agents=killed_agents)
			# 		episode_gui(env, w1, w2)
			# 		episode_gui(env, w1, w2)
			env.reset()
			env.reset_to_custom_line_formation(xs, 20)
			# env.reset_to_line_formation()
			# env.reset_to_unsimmetric_line_formation()
			for i in killed_agents:
				env.agents[i].is_dead = True
			env.reset_leader()
			# S=0
			old_xl=env.xL
			old_yl=env.yL
			first_xl=env.xL
			first_yl = env.yL
			np_d[n1-1, 0]=0
			np_angle[n1-1, 0]=0
			t=0
			while not env.is_done:
				t+=1
				env.play_step_only_formation(w1)
				np_d[n1-1, t]=getDistance(env.xL, env.yL, old_xl, old_yl)
				np_angle[n1-1, 0]=segmentAngleWithXAxis( old_xl, old_yl,env.xL, env.yL)
				old_xl = env.xL
				old_yl = env.yL

			aac = alive_agent_count(env)
			lgd = leader_goal_distance(env)
			row = [w1,  alive_agent_count(env), numpy.sum(np_d[n1-1, :]),getDistance(env.xL, env.yL, first_xl, first_yl),
			       segmentAngleWithXAxis( first_xl, first_yl,env.xL, env.yL), env.t, meanV(env.v_history),
			       env.tForm,  n2]
			df.loc[run] = row
			print(row)
			# if aac < 9 - len(killed_agents) or not equals(lgd, 0.0):
			# 	raise Exception(str(row) + str(killed_agents))
			env.save_episode(os.path.join(directory, str(run)))

			np1[n1 - 1, n2] = row[2]
			np2[n1 - 1, n2] = row[3]
			np3[n1 - 1, n2] = row[4]
			np4[n1 - 1, n2] = row[7]


			run = run + 1

		W = numpy.outer(numpy.array(range(1, 101)) / 10, numpy.ones(TMax))
		T = numpy.outer(numpy.ones(100), numpy.array(range(TMax)))
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		ax.azim = 60
		ax.set(
			xlabel='Вес скорости сохранения формы(W1)', ylabel='Шаг', zlabel='Растояние на которую сдвинулся виртуальный лидер'			)
		ax.plot_surface(W, T, np_d, cmap='viridis', edgecolor='green')

		fig.savefig(os.path.join(child_dir, 'dist.png'))
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		ax.azim = 60
		ax.set(
			xlabel='Вес скорости сохранения формы(W1)', ylabel='Шаг', zlabel='Угол движения виртуального лидера'
		)
		ax.plot_surface(W, T, np_angle, cmap='viridis', edgecolor='green')

		fig.savefig(os.path.join(child_dir, 'angle.png'))
		plt.close('all')

	writer = pandas.ExcelWriter(os.path.join(directory, 'report.xls'), engine='xlsxwriter')

	df.to_excel(writer, sheet_name='report')
	writer.close()
	W = numpy.array(range(1, 101)) / 10
	# k=  range(9)
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[2])
	for i in range(9):
		ax.plot(W, np1[:, i], label=str(i) + ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[2] + '_1.png'))

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[3])
	for i in range(9):
		ax.plot(W, np2[:, i], label=str(i) + ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[3] + '_2.png'))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[4])
	for i in range(9):
		ax.plot(W, np3[:, i], label=str(i) + ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[4] + '_3.png'))
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set(xlabel=df.columns[0] + '(w1)', ylabel=df.columns[7])
	for i in range(9):
		ax.plot(W, np4[:, i], label=str(i) + ' crashed')
	ax.legend()
	fig.savefig(os.path.join(img_directory, df.columns[7] + '_4.png'))

	W1 = numpy.outer(numpy.array(range(1, 101)) / 10, numpy.ones(9))
	K = numpy.outer(numpy.ones(100), range(9))

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.azim = 60
	ax.set(xlabel=df.columns[1] + '(w1)', ylabel=df.columns[8], zlabel=df.columns[3])
	ax.plot_surface(W1, K, np1, cmap='viridis', edgecolor='green')
	fig.savefig(os.path.join(img_directory, df.columns[3] + '3d.png'))

if __name__=='__main__':
	# plot_only_formation()
	# plot_leader_goal_distance()
	env = Env(
		width=100, height=100, goal_x=50, goal_y=80, N=ROBOT_NUMBER,
		desired_X=DX, desired_Y=DY, leader_x=50, leader_y=20, robot_radius=ROBOT_RADIUS,
		buffer_size=MAX_T
	)
	episode_gui(env,1, 1, paused=True)