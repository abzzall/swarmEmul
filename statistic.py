import os

import numpy
from matplotlib import cm

from constants import *
from env import Env
import pandas
import matplotlib
import matplotlib.pyplot as plt

from geometry import equals, getDistance
from visualiser import episode_gui


def desiredXYSquarePattern(N: int, d=SENSOR_RANGE):
    X = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N)] * N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y

def leader_goal_distance(env: Env):
	return getDistance(env.xL, env.yL, env.xG, env.yG)
if __name__ == '__main__':
    N = 2
    sensor_range = 20
    ROBOT_RADIUS=1
    Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
    env = Env(500, 500, 250,450, N * N, Dx, Dy, SENSOR_RANGE, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    env.addObstacle(200, 200, 300, 300)
    df = pandas.DataFrame(columns=['id', 'Вес скорости избегания столкновении',
                                   'Вес скорости сохранения формы', 'Вес скорости достижения цели',
                                   'Количество выживших агентов', 'Количество итерации',
                                   ])
    df.set_index('id', inplace=True)
    # episode_gui_(5, 1, 1)
    # N = 2
    # sensor_range = 20
    # ROBOT_RADIUS = 1
    # Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
    # env = Env(500, 500, 250, 450, N * N, Dx, Dy, SENSOR_RANGE, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    # env.addObstacle(200, 200, 300, 300)
    # env.addObstacle(60, 80, 40, 120)

    # env.addObstacle(350, 200, 400, 250)
    # env.addObstacle(100, 200, 150, 250)
    # episode_gui(env,15, 1, 1, draw_way=True)
    W = range(20)
    run = 0
    dir_name = os.path.join('..', 'run')
    for w1 in W:
        print('w1=' + str(w1))
        for w2 in W:
            print('w2=' + str(w2))
            for w3 in W:
                print('w3=' + str(w3))
                # history['v'], history['pose'], history['angle'], history['detection'], history['dead'], \
                # history['width'], \
                # history['height'], history['goal_x'], history['goal_y'], history['wall'], history['radius'], history[
                #     'dx'], history['dy'], history['N'], history['t'], history['leader_x'], history['leader_y']
                ep_file=os.path.join(dir_name, str(run)+'_'+str(w1)+'_'+str(w2)+'.npz')
                # v, pose, angle, detection, dead, width, height, goalX, goalY, wall, radius, dX, dY, N, t, lX, lX=Env.load_episode_history(ep_file)
                episode_gui(env, 19, 18, 2)
                # env.episode(w1, w2, w3)
                env.save_episode(os.path.join(dir_name, str(run)+'_'+str(w1)+'_'+str(w2)))
                alive_agent_count=N*N-numpy.sum(env.dead_history[env.t-1,:])
                t=env.t if (equals(leader_goal_distance(env), 0.0)) else None
                row=[w1, w2, w3, alive_agent_count, t]
                print(row)
                df.loc[run] = row
                run = run + 1

    df=df.apply(pandas.to_numeric, errors='coerce')
    df.to_excel(os.path.join(dir_name, 'report.xlsx'))
    df['Вес скорости избегания столкновении']=df['Вес скорости избегания столкновении'].astype(float)
    df['Вес скорости сохранения формы']=df['Вес скорости сохранения формы'].astype(float)
    df['Вес скорости достижения цели']=df['Вес скорости достижения цели'].astype(float)
    df['Количество выживших агентов']=df['Количество выживших агентов'].astype(float)
    df['Количество итерации']=df['Количество итерации'].astype(float)
    for w in W:
        df1=df[df['Вес скорости избегания столкновении']==w]
        df2=df[df['Вес скорости сохранения формы']==w]
        df3=df[df['Вес скорости достижения цели']==w]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df1['Вес скорости сохранения формы'], df1['Вес скорости достижения цели'], df1['Количество выживших агентов'], cmap=cm.jet, linewidth=0.2)
        fig.savefig(os.path.join(dir_name, 'stat', 'quantity', 'w1', str(w)+'.png'))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df1['Вес скорости сохранения формы'], df1['Вес скорости достижения цели'], df1['Количество итерации'], cmap=cm.jet, linewidth=0.2)
        fig.savefig(os.path.join(dir_name, 'stat', 'time', 'w1', str(w)+'.png'))



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df2['Вес скорости избегания столкновении'], df2['Вес скорости достижения цели'],
                        df2['Количество выживших агентов'], cmap=cm.jet, linewidth=0.2)
        fig.savefig(os.path.join(dir_name, 'stat', 'quantity', 'w2', str(w) + '.png'))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df2['Вес скорости избегания столкновении'], df2['Вес скорости достижения цели'],
                        df2['Количество итерации'], cmap=cm.jet, linewidth=0.2)
        fig.savefig(os.path.join(dir_name, 'stat', 'time', 'w2', str(w) + '.png'))



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df3['Вес скорости сохранения формы'], df3['Вес скорости избегания столкновении'],
                        df3['Количество выживших агентов'], cmap=cm.jet, linewidth=0.2)
        fig.savefig(os.path.join(dir_name, 'stat', 'quantity', 'w3', str(w) + '.png'))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df3['Вес скорости сохранения формы'], df3['Вес скорости избегания столкновении'],
                        df3['Количество итерации'], cmap=cm.jet, linewidth=0.2)
        fig.savefig(os.path.join(dir_name, 'stat', 'time', 'w3', str(w) + '.png'))

        #
        # df1.plot(x='Вес скорости сохранения формы', y='Вес скорости достижения цели', z='Количество выживших агентов').get_figure().savefig(os.path.join(dir_name, 'stat', 'quantity', 'w1', str(w)+'.png'))
        #
        # df1.plot(x='Вес скорости сохранения формы', y='Вес скорости достижения цели', z='Количество итерации').get_figure().savefig(os.path.join(dir_name, 'stat', 'time', 'w1', str(w)+'.png'))
        #
        #
        # df2.plot(x='Вес скорости избегания столкновении', y='Вес скорости достижения цели', z='Количество выживших агентов').get_figure().savefig(os.path.join(dir_name, 'stat', 'quantity', 'w2', str(w)+'.png'))
        #
        # df2.plot(x='Вес скорости избегания столкновении', y='Вес скорости достижения цели', z='Количество итерации').get_figure().savefig(os.path.join(dir_name, 'stat', 'time', 'w2', str(w)+'.png'))
        #
        # df3.plot(x='Вес скорости сохранения формы', y='Вес скорости избегания столкновении', z='Количество выживших агентов').get_figure().savefig(os.path.join(dir_name, 'stat', 'quantity', 'w3', str(w)+'.png'))
        #
        # df3.plot(x='Вес скорости сохранения формы', y='Вес скорости избегания столкновении', z='Количество итерации').get_figure().savefig(os.path.join(dir_name, 'stat', 'time', 'w3', str(w)+'.png'))



