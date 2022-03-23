import numpy
import pandas

from constants import *



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
from without_form.env_without_form import Env
from without_form.visualiser import episode_gui

def leader_goal_distance(env: Env):
    return getDistance(env.xL, env.yL, env.xG, env.yG)
def desiredXYSquarePattern(N: int, d=SENSOR_RANGE):
    X = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N)] * N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y
if __name__ == '__main__':
    N = 3
    sensor_range = 5
    ROBOT_RADIUS = 1
    Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
    env = Env(200, 200, 100, 180, N * N, Dx, Dy, SENSOR_RANGE, 100, 20, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    env.addObstacle(80, 80, 120, 120)
    episode_gui(env, 1, 0.3, draw_way=False)
    alive_agent_count = N * N - numpy.sum(env.dead_history[env.t - 1, :])
    t = env.t if (equals(leader_goal_distance(env), 0.0)) else None
    print(t)
    print(alive_agent_count)