import constants
from constants import ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T
from env import Env
import numpy

from main import alive_agent_count
from visualiser import episode_gui


def desiredXYSquarePattern(N: int, d=constants.SENSOR_RANGE):
    X=numpy.array([(i-(N-1)*0.5)*d for i in range(N)]*N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y

def main():
    N=10
    sensor_range=5
    Dx, Dy=desiredXYSquarePattern(N, sensor_range+ROBOT_RADIUS)
    env=Env(500, 500, 250, 450, N*N, (200, 200, 300, 300),Dx, Dy,  sensor_range, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    episode_gui(env, 5, 0.1, 0.1)
    print(alive_agent_count(env))

if __name__ == '__main__':
    main()