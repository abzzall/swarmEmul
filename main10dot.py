import constants
from constants import ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T
from env import Env
import numpy
import os
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
    env=Env(500, 500, 250, 450, N*N, Dx, Dy,  sensor_range, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    env.addObstacle(225, 100, 275, 150)
    env.addObstacle(350, 200, 400, 250)
    env.addObstacle(100, 200, 150, 250)
    episode_gui(env, 5, 0.3, 0.3)
    print(alive_agent_count(env))
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
def kill_list(n):
	p=[perebor(0, n-1, i) for i in range(n)]
	s=1
	i=0
	killed_list=[]
	while s>0:
		s=0
		for l in p:
			if len(l)>i:
				killed_list=killed_list+[l[i]]
				s+=1
				print(l[i])
		i+=1
	return killed_list

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


if __name__ == '__main__':
	N=10
	sensor_range = 5
	Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
	env = Env(500, 500, 250, 450, N * N, Dx, Dy, sensor_range, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
	env.addObstacle(225, 100, 275, 150)
	env.addObstacle(350, 200, 400, 250)
	env.addObstacle(100, 200, 150, 250)

	n=N*N
	directory='..\\run1dot'
	file_name=os.path.join(directory, 'var.npz')
	if os.path.exists(file_name):
		cur_state=numpy.load(file_name)
		killed_list=cur_state['killed']
		n01=int(cur_state['n1'])
		n02 = int( cur_state['n2'])
		n03 = int( cur_state['n3'])
		run=int( cur_state['run'])
	else:
		killed_list=[]
		n01=100
		n02 = 100
		n03 =100
		run=0

	while not(killed_list is None):
		print(killed_list)
		for n2 in range(n02, -1, -1):
			w2=n2/100
			print('w2=',w2)
			for n3 in range(n03, -1, -1):
				w3=n3/100
				print('w3=', w3)
				for n1 in range(n01, -1, -1):
					w1=n1/100
					print('w1=', w1)
					print('run='+str(run))
					ep_file=os.path.join(directory, str(run))

					env.episode(w1, w2, w3, killed_list)
					env.save_episode(ep_file)

					# env.episode(w1/10, w2/10, w3/10, killed_list)
					# env.save_episode(ep_file+'a')

					numpy.savez(file_name, killed=killed_list, run=run, n1=n1, n2=n2, n3=n3)
					run=run+1
				n01=100
			n03=100
		n02=100

		killed_list=nextList(n, killed_list)


