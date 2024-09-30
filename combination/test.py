import csv
import math
import os
from math import pi

import numpy
import pandas

from combination.constants import *



# if __name__ == '__main__':
#
#
#     # episode_gui_(5, 1, 1)
#     N = 2
#     sensor_range = 20
#     ROBOT_RADIUS=1
#     Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
#     env = Env(500, 500, 250,450, N * N, Dx, Dy, SENSOR_RANGE, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
#     env.addObstacle(200, 200, 300, 300)
#     # env.addObstacle(60, 80, 40, 120)
#
#     # env.addObstacle(350, 200, 400, 250)
#     # env.addObstacle(100, 200, 150, 250)
#     # episode_gui(env,15, 1, 1, draw_way=True)
#     W=range(20)
#     run=0
#     dir_name=os.path.join('..', 'run')
#     for w1 in W:
#         print('w1='+str(w1))
#         for w2 in W:
#             print('w2='+str(w2))
#             for w3 in W:
#                 print('w3=' + str(w3))
#                 env.episode(w1, w2, w3)
#                 env.save_episode(os.path.join(dir_name, str(run)+'_'+str(w1)+'_'+str(w2)))
#                 run=run+1


# env.save_episode('test')
# episode_replay_from_file('test.npz', WINDOW_SIZE, WINDOW_SIZE)
from geometry import equals, getDistance
from env_combined import Env
from combination.visualiser import episode_gui

def leader_goal_distance(env: Env):
    return getDistance(env.xL, env.yL, env.xG, env.yG)
def desiredXYSquarePattern(N: int, d=SENSOR_RANGE):
    X = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N)] * N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y
if __name__ == '__main__':
    # W= numpy.array([1, 2, 3, 5, 10, 15, 30, 45, 60])
    # V=numpy.array(range(1, 51))/10
    # directory = '..\\..\\run_complex'
    # file_name = os.path.join(directory, 'var.npz')
    # report_file_name=os.path.join(directory, 'report.xlsx')
    # if os.path.exists(file_name):
    #     cur_state = numpy.load(file_name)
    #
    #     w0i = int(cur_state['w'])
    #     v0i = int(cur_state['v'])
    #     run = int(cur_state['run'])
    # else:
    #     w0i=0
    #     v0i=0
    #     run = 0
    N = 3
    sensor_range = 3
    # ROBOT_RADIUS = 0.3
    Dx, Dy = desiredXYSquarePattern(N, sensor_range+ROBOT_RADIUS)
    #############################################
    env = Env(250, 250, 210, 160, N * N, Dx, Dy, sensor_range, 20, 20, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, 5000000, large_sensor_range=20)
    # env.addObstacle(50, 0, 60, 200)
    env.addObstacle(50, 0, 60, 80)
    env.addObstacle(50, 100, 60, 220)

    env.addObstacle(100, 40, 220, 50)

    env.addObstacle(100, 50, 110, 180)
    env.addObstacle(100, 200, 110, 250)

    env.addObstacle(170, 100, 180, 220)

    env.addObstacle(200, 100, 250, 110)

    ###########################

    # env = Env(250, 250, 125, 220, N * N, Dx, Dy, sensor_range, 125, 20, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, 5000000)
    # env.addObstacle(100, 70, 150, 100)
    #
    # env.addObstacle(50, 130, 100, 170)
    # env.addObstacle(150, 130, 200, 170)
    ############################################


    v=0.1
    w=30
    episode_gui(env, 0.3, 0.1, 0.5, draw_way=True, window_width=1000, window_height=1000, fps=150)
    # # env.addObstacle(50, 80, 150, 120)
    # row = ['row','v', 'w', 't', 'alive_agent_count']
    # with open(report_file_name, 'a') as report_file:
    #     writer = csv.writer(report_file)
    #     writer.writerow(row)
    #
    # for wi in range(w0i, W.size):
    #     w=W[wi]
    #     print(w)
    #     for vi in range(v0i,  V.size):
    #         v=V[vi]
    #         env.reset()
    #         env.agents[0].left=env.agents[3]
    #         env.agents[1].left=env.agents[0]
    #         env.agents[2].left=env.agents[1]
    #         env.agents[5].left=env.agents[2]
    #         env.agents[8].left=env.agents[5]
    #         env.agents[7].left=env.agents[8]
    #         env.agents[6].left=env.agents[7]
    #         env.agents[3].left=env.agents[6]
    #
    #
    #         env.agents[3].right= env.agents[0]
    #         env.agents[0].right= env.agents[1]
    #         env.agents[1].right= env.agents[2]
    #         env.agents[2].right= env.agents[5]
    #         env.agents[5].right= env.agents[8]
    #         env.agents[8].right= env.agents[7]
    #         env.agents[7].right= env.agents[6]
    #         env.agents[6].right= env.agents[3]
    #
    #         for i in range(1, N-1):
    #             for j in range(1, N-1):
    #                 env.agents[i*N+j].sensored=False
    #         while(not env.is_done):
    #             env.play_step(v,w*pi/180)
    #         env.save_episode(os.path.join(directory, 'run'+str(run)))
    #         alive_agent_count = N * N - numpy.sum(env.dead_history[env.t - 1, :])
    #         t = env.t if (equals(leader_goal_distance(env), 0.0)) else None
    #         # print(t)
    #         # print(alive_agent_count)
    #         row=[run, v, w, t, alive_agent_count]
    #         print(row)
    #         with open(report_file_name, 'a') as report_file:
    #             writer=csv.writer(report_file)
    #             writer.writerow(row)
    #         numpy.savez(file_name, w=wi, run=run, v=vi)
    #         run+=1
    print(env.t)


