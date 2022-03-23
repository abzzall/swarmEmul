import os

import numpy
import pandas

from constants import *
from env import Env


def nextList(n, l=None):
	if l is None:
		return [];
	elif len(l)==n:
		return None
	s=len(l)
	cur=s-1

	while(cur>=0):
		if((cur<s-1 and l[cur]+1<l[cur+1]) or( cur==s-1 and  l[cur]<n-1) ):
			new_l=l.copy()
			for i in range(cur, s):
				new_l[i]=l[cur]+i-cur+1
			return new_l
		cur=cur-1
	return [*range(s+1)]
def desiredXYSquarePattern(N: int, d=SENSOR_RANGE):
    X=numpy.array([(i-(N-1)*0.5)*d for i in range(N)]*N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y

def main():
    df = pandas.DataFrame(columns=['id',  'Вес скорости избегания столкновении',
                                   'Вес скорости сохранения формы', 'Вес скорости достижения цели',
                                   'Количество выживших агентов',
                                   'Расстояние между виртуальным лидером и целевой точки', 'Количество итерации',
                                   'Средний скорость'])
    df.set_index('id', inplace=True)
    N = 10
    sensor_range = 5
    Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
    env = Env(500, 500, 250, 450, N * N, Dx, Dy, sensor_range, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    env.addObstacle(225, 100, 275, 150)
    env.addObstacle(350, 200, 400, 250)
    env.addObstacle(100, 200, 150, 250)

    n = N * N
    directory = '..\\run15'
    killed_list = []

    W=[*range(10, -1, -2)]
    run=0
    done=False
    while not (killed_list is None):
        print(killed_list)
        for n2 in range(10, -1, -2):
            w2 = n2
            print('w2=', w2)
            for n3 in range(10, -1, -2):
                w3 = n3
                print('w3=', w2)
                for n1 in range(10, -1, -2):
                    w1=n1
                    ep_file = os.path.join(directory, str(run))
                    if not os.path.exists(ep_file):
                        return df

                    #todo

                    run=run+1
        killed_list = nextList(n, killed_list)
    return df

