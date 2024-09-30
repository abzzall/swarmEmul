import numpy

from dif_sensor.constants import *
from dif_sensor.env import Env
from dif_sensor.visualiser import episode_gui


def desiredXYSquarePattern(N: int, d=SENSOR_RANGE):
    X = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N)] * N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y


if __name__ == '__main__':
    # episode_gui_(5, 1, 1)
    N = 5
    sensor_range = 10
    ROBOT_RADIUS=1
    Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
    env = Env(500, 500, 250, 450, N * N, Dx, Dy, sensor_range, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    env.addObstacle(225, 100, 300, 150)
    # env.addObstacle(300, 50, 350, 100)
    # env.addObstacle(100, 200, 150, 250)

    episode_gui(env,1, 0.1, 0.1)
# env.save_episode('test')
# episode_replay_from_file('test.npz', WINDOW_SIZE, WINDOW_SIZE)

# if __name__ == '__main__':
#     # episode_gui_(5, 1, 1)
#     N = 5
#     sensor_range = 5
#     ROBOT_RADIUS=1
#     Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
#     env = Env(500, 500, 250,450, N * N, Dx, Dy, SENSOR_RANGE, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
#     env.addObstacle(200, 200, 300, 300)
#     # env.addObstacle(60, 80, 40, 120)
#
#     # env.addObstacle(350, 200, 400, 250)
#     # env.addObstacle(100, 200, 150, 250)
#     episode_gui(env,5, 2, 1, draw_way=True)