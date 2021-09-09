from typing import List
from geometry import *
import numpy
from numpy import pi


UNIT=1
ROBOT_RADIUS=UNIT/2
SENSOR_RANGE= 3
SENSOR_DETECTION_COUNT = 12

XL=50
YL=15


VIZ_SIZE=200
MIN_RADIUS=5

ROBOT_NUMBER=9
DX=[-SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -ROBOT_RADIUS - SENSOR_RANGE, 0, ROBOT_RADIUS + SENSOR_RANGE];
DY=[-SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, 0, 0, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS];


GOAL_X=50
GOAL_Y=90

OBSTACLE_POS=(40, 40, 60, 60)

W1=5
W2=1
W3=1

def lidar_angles(sensor_detection_count=SENSOR_DETECTION_COUNT):
	return [
		-pi+i*2*pi/sensor_detection_count
		for i in
		range(sensor_detection_count)]
class Action:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def v(self):
		return sqrt(self.x**2+self.y**2)
	def active(self):
		return self.v()>0.001

	def __add__(self, other):
		return Action(self.x + other.x, self.y + other.y)

	def __rmul__(self, other):
		return Action(self.x *other, self.y * other)
	def __str__(self):
		return '('+str(self.x)+', '+str(self.y)+')'

	def __eq__(self, other):
		return self.x==other.x and self.y==other.y

class Visualizer:
	def __init__(self, width=VIZ_SIZE, height=VIZ_SIZE, xG=GOAL_X, yG=GOAL_Y, N=ROBOT_NUMBER, obstacle_pos=OBSTACLE_POS, Dx=DX, Dy=DY, sensor_range=SENSOR_RANGE, xL=XL, yL=YL, robot_radius=ROBOT_RADIUS, sensor_detection_count=SENSOR_DETECTION_COUNT):
		# pygame.init()
		self.width = width
		self.height = height

		self.xG=xG
		self.yG=yG
		self.obstacle_pos=obstacle_pos
		self.N=N
		self.Dx=Dx
		self.Dy=Dy
		self.sensor_range=sensor_range
		self.xL=xL
		self.yL=yL
		self.robot_radius=robot_radius
		self.sensor_detection_count=sensor_detection_count

	def reset(self):
		self.isDone = False
		self.agents = []
		self.walls = []
		# external walls
		wall1 = Wall(0, 0, 1, self.height)
		wall2 = Wall(0, self.height-1 , self.width, self.height)
		wall3 = Wall(self.width-1, 0, self.width, self.height)
		wall4 = Wall(0, 0, self.width-1, 1)

		from_x, from_y, to_x, to_y=self.obstacle_pos
		obstacle=Wall(from_x, from_y, to_x, to_y)


		self.walls.append(wall1)
		self.walls.append(wall2)
		self.walls.append(wall3)
		self.walls.append(wall4)
		self.walls.append(obstacle)

		for i in range(self.N):
			agent=Agent(self.xL + self.Dx[i], self.yL + self.Dy[i], self.Dx[i], self.Dy[i], radius=self.robot_radius, sensor_detection_count=self.sensor_detection_count)
			self.agents.append(agent)
		self.t=0


	def play_step(self, w1, w2, w3):
		# for t in range(FPS):
		self.observe()
		moving=False
		self.t+=1
		for i in range(self.N):
			agent = self.agents[i]
			if agent.isdead:
				continue
			v1=vAvoidObsMinAngle(agent, self.sensor_range)
			v2=vKeepFormation(agent, self.xL, self.yL,  w2)
			v3=vGoal(self.xL, self.yL, self.xG, self.yG, w3)
			v=w1*v1+v2+v3
			if self.old_v[agent.id]!=v or self.old_v1[agent.id]!=v1 or self.old_v2[agent.id]!=v2 or self.old_v3[agent.id]!=v3:
				print(' t='+str(self.t)+'v1='+str(v1)+' v2='+str(v2)+' v3='+str(v3)+' v='+str(v))
				self.old_v[agent.id] = v
				self.old_v1[agent.id] = v1
				self.old_v2[agent.id] = v2
				self.old_v3[agent.id] = v3
			agent.move(v)
			if v.v()<0.1:
				print('v1='+str(v1)+' v2='+str(v2)+' v3='+str(v3)+' v='+str(v))
			if v.active():
				moving=True



		if not moving:
			self.isDone=True

		self.checkDead()

		self.resetLeader()

	def resetLeader(self):
		self.xL, self.yL=virtual_leader_position(self.agents)

	def checkDead(self):
		for i in range(self.N):
			agent = self.agents[i]
			if agent.isdead:
				continue

			for drawable in self.agents:
				if drawable != agent and agent.isCollide(drawable):
					agent.isdead=True
					drawable.isdead=True
					break
			if not agent.isdead:
				for drawable in self.walls:
					if drawable != agent and agent.isCollide(drawable):
						agent.isdead = True
						break



	def observe(self):
		for i in range(self.N):
			agent = self.agents[i]
			agent.detected=False
			for j,angle in enumerate(agent.get_lidar_angles()):
				min_dist = self.sensor_range + 1
				for drawable in self.agents+self.walls:
					if drawable != agent:
						# nn=agent.get_absolute_angle(angle)
						dist = drawable.getIntersection(
							x=agent.x, y=agent.y,
							angle=angle
							)
						if dist == -1:
							continue
						dist = max(dist, 0)
						min_dist = min(dist, min_dist)
				if min_dist <= self.sensor_range:
					agent.obs[j]=min_dist
					agent.detected=True
				else:
					agent.obs[j]=0

	def update_agent_distances(self):
		self.agent_distances = [[0] * self.N] * self.N
		for i in range(self.N):
			for j in range(i + 1, self.N):
				d = self.agents[i].min_distance(self.agents[j])
				self.agent_distances[i][j] = d
				self.agent_distances[j][i] = d

class Drawable():
	def __init__(self, x, y, lengthX=UNIT, lengthY=UNIT):
		self.x = x
		self.y = y
		self.lengthX = lengthX
		self.lengthY = lengthY
		self.update_pos()

	def update_pos(self):
		self.fromX = self.x - self.lengthX / 2
		self.fromY = self.y - self.lengthY / 2
		self.toX = self.x + self.lengthX / 2
		self.toY = self.y + self.lengthY / 2

	def containsPoint(self, x, y):
		return self.fromX <= x <= self.toX and self.fromY <= y <= self.toY

	def isCollide(self, drawable):
		return self.containsPoint(drawable.fromX, drawable.fromY) \
		       or self.containsPoint(drawable.fromX, drawable.toY) \
		       or self.containsPoint(drawable.toX, drawable.toY) \
		       or self.containsPoint(drawable.toX, drawable.fromY) \
		       or drawable.containsPoint(self.fromX, self.fromY) \
		       or drawable.containsPoint(self.fromX, self.toY) \
		       or drawable.containsPoint(self.toX, self.toY) \
		       or drawable.containsPoint(self.toX, self.fromY) \
		       or (self.fromX <= drawable.fromX and self.toX >= drawable.toX and self.fromY >= drawable.fromY and \
		           self.toY <= drawable.toY) \
		       or (self.fromX >= drawable.fromX and self.toX <= drawable.toX and self.fromY <= drawable.fromY and \
		           self.toY >= drawable.toY)

	def isCollideRect(self, fromX, fromY, toX, toY):
		return self.containsPoint(fromX, fromY) \
		       or self.containsPoint(fromX, toY) \
		       or self.containsPoint(toX, toY) \
		       or self.containsPoint(toX, fromY) \
		       or (fromX <= self.fromX <= toX and fromY <= self.fromX <= toY) \
		       or (fromX <= self.fromY <= toX and fromY <= self.fromY <= toY) \
		       or (fromX <= self.toX <= toX and fromY <= self.toX <= toY) \
		       or (fromX <= self.toY <= toX) and (fromY <= self.toY <= toY) \
		       or (self.fromX <= fromX and self.toX >= toX and self.fromY >= fromY and \
		           self.toY <= toY) \
		       or (self.fromX >= fromX and self.toX <= toX and self.fromY <= fromY and \
		           self.toY >= toY)

	def isIntersect(self, x1, y1, x2, y2):
		if (x1 < self.fromX and x2 < self.fromX) or (x1 > self.toX and y1 > self.toX) \
				or (y1 < self.fromY and y2 < self.toY) \
				or (y1 > self.toY and y2 > self.toY):
			return False
		angle1 = calculateAngle(self.fromX, self.fromY, x1, y1, x2, y2)
		angle2 = calculateAngle(self.toX, self.fromY, x1, y1, x2, y2)
		angle3 = calculateAngle(self.fromX, self.toY, x1, y1, x2, y2)
		angle4 = calculateAngle(self.toX, self.toY, x1, y1, x2, y2)

		if ((angle1 > 0 and angle2 > 0 and angle3 > 0 and angle4 > 0) or (
				angle1 < 0 and angle2 < 0 and angle3 < 0 and angle4 < 0)):
			return False
		return True

	def getIntersection(self, x, y, angle):
		if self.containsPoint(x, y):
			return 0
		if angle > pi or angle < -pi:
			return self.getIntersection(x, y, normAngleMinusPiPi(angle))
		if angle == -pi or angle == pi:
			if self.toX <= x and self.fromY <= y <= self.toY:
				return x - self.toX
			else:
				return -1
		elif angle == -pi / 2:
			if self.toY < y and self.fromX < x < self.toX:
				return y - self.toY
			else:
				return -1
		elif angle == 0:
			if self.fromX > x and self.fromY < y < self.toY:
				return self.fromX - x
			else:
				return -1
		elif angle == pi / 2:
			if self.fromY > y and self.fromX < x < self.toX:
				return self.fromY - y
			else:
				return -1
		k = tan(angle)
		b = y - k * x
		if angle < -pi / 2:
			if self.toX < x:
				yIntersection = k * self.toX + b
				if self.fromY <= yIntersection <= self.toY:
					return getDistance(x, y, self.toX, yIntersection)
			if self.toY < y:
				xIntersection = (self.toY - b) / k
				if self.fromX <= xIntersection <= self.toX:
					return getDistance(x, y, xIntersection, self.toY)
		elif -pi / 2 < angle < 0:
			if self.toY < y:
				xIntersection = (self.toY - b) / k
				if self.fromX <= xIntersection <= self.toX:
					return getDistance(x, y, xIntersection, self.toY)
			if self.fromX > x:
				yIntersection = k * self.fromX + b
				if self.fromY <= yIntersection <= self.toY:
					return getDistance(x, y, self.fromX, yIntersection)
		elif 0 < angle < pi / 2:
			if self.fromX > x:
				yIntersection = k * self.fromX + b
				if self.fromY <= yIntersection <= self.toY:
					return getDistance(x, y, self.fromX, yIntersection)
			if self.fromY > y:
				xIntersection = (self.fromY - b) / k
				if self.fromX <= xIntersection <= self.toX:
					return getDistance(x, y, xIntersection, self.fromY)
		elif angle > pi / 2:
			if self.toX < x:
				yIntersection = k * self.toX + b
				if self.fromY <= yIntersection <= self.toY:
					return getDistance(x, y, self.toX, yIntersection)
			if self.fromY > y:
				xIntersection = (self.fromY - b) / k
				if self.fromX <= xIntersection <= self.toX:
					return getDistance(x, y, xIntersection, self.fromY)
		return -1

	def min_distance(self, drawable):

		if drawable.fromX <= self.fromX <= drawable.toX or drawable.fromX <= self.toX <= drawable.toX or \
				self.fromX <= drawable.fromX <= self.toX or self.fromX <= drawable.toX <= self.toX:
			minX = 0
		else:
			minX = min(
				abs(self.fromX - drawable.fromX)
				, abs(self.fromX - drawable.toX)
				, abs(self.toX - drawable.fromX)
				, abs(self.toX - drawable.toX)
			)
		if drawable.fromY <= self.fromY <= drawable.toY or drawable.fromY <= self.toY <= drawable.toY or \
				self.fromY <= drawable.fromY <= self.toY or self.fromY <= drawable.toY <= self.toY:
			minY = 0
		else:
			minY = min(
				abs(self.fromY - drawable.fromY)
				, abs(self.fromY - drawable.toY)
				, abs(self.toY - drawable.fromY)
				, abs(self.toY - drawable.toY)
			)
		if minX == 0:
			return minY
		if minY == 0:
			return minX
		return sqrt(minX ** 2 + minY ** 2)

	@property
	def fromX(self):
		if self._fromX is None:
			self._fromX = self.x - self.lengthX / 2
		return self._fromX

	@property
	def fromY(self):
		if self._fromY is None:
			self._fromY = self.y - self.lengthX / 2
		return self._fromY

	@property
	def toX(self):
		if self._toX is None:
			self._toX = self.x + self.lengthX / 2
		return self._toX

	@property
	def toY(self):
		if self._toY is None:
			self._toY = self.y + self.lengthX / 2
		return self._toY

	@fromX.setter
	def fromX(self, value):
		self._fromX = value

	@fromY.setter
	def fromY(self, value):
		self._fromY = value

	@toX.setter
	def toX(self, value):
		self._toX = value

	@toY.setter
	def toY(self, value):
		self._toY = value

class Agent(Drawable):
	id=0
	def __init__(self,  x, y,dx, dy,  angle=0, radius=ROBOT_RADIUS, sensor_detection_count=SENSOR_DETECTION_COUNT):
		self.angle=angle
		self.isdead = False
		self.sensor_detection_count=sensor_detection_count
		self.obs=numpy.zeros(self.sensor_detection_count)
		self.detected=False
		self.dx=dx
		self.dy=dy
		self.id=Agent.id
		Agent.id+=1
		self.radius=radius
		Drawable.__init__(self, x, y, lengthX=self.radius*2, lengthY=self.radius*2)
		self.update_pos()

	def movePolar(self, step_size, angle):
		self.angle=angle
		self.x += step_size * cos(self.angle)
		self.y += step_size * sin(self.angle)
		self.update_pos()
	def move(self, action:Action):
		self.x+=action.x
		self.y+=action.y
		if action.active():
			self.angle=angleWithXAxis(action.x, action.y)
		self.update_pos()

	def moveTo(self, newX, newY):
		if newX==self.x and newY==self.y:
			pass
		angle=segmentAngleWithXAxis(self.x, self.y, newX, newY)
		self.x=newX
		self.y=newY
		self.angle = angle
		self.update_pos()

	def get_distance(self, drawable):
		return sqrt((self.x - drawable.x) ** 2 + (self.y - drawable.y) ** 2) - self.radius

	def get_rel_angle(self, drawable):
		if drawable.x == self.x:
			if drawable.y == self.y:
				return 0
			elif drawable.y > self.y:
				return pi * 0.5
			else:
				return -pi * 0.5
		return atan((drawable.y - self.y) / (drawable.x - self.x))

	def get_absolute_angle(self, angle):
		return normAngleMinusPiPi(angle + self.angle)
	def get_lidar_angles(self):
		return [ self.get_absolute_angle(angle) for angle in lidar_angles(self.sensor_detection_count) ]


class Wall(Drawable):
	def __init__(self, fromX, fromY, toX, toY):
		w=toX - fromX
		h=toY - fromY
		Drawable.__init__(
			self, x=(toX + fromX) / 2, y=(toY + fromY) / 2, lengthX=w,
			lengthY=h
			)


def virtual_leader_position(agents: List[Agent]):
	x=[]
	y=[]
	dx=[]
	dy=[]
	for agent in agents:
		if not agent.isdead:
			x.append(agent.x)
			y.append(agent.y)
			dx.append(agent.dx)
			dy.append(agent.dy)
	x=numpy.array(x)
	y=numpy.array(y)
	dx=numpy.array(dx)
	dy=numpy.array(dy)
	return numpy.mean(x-dx), numpy.mean(y-dy)

def vAvoidObsMinAngle(agent: Agent, sensor_range)->Action:
	if not agent.detected:
		return Action(0, 0)

	angles=numpy.array(agent.get_lidar_angles())
	agent.obs[agent.obs==0]=sensor_range
	nom=0
	denom=0
	for i in range(len(angles)):
		nom+=angles[i]*agent.obs[i]
		denom+=agent.obs[i]

	turn_angle=nom/denom
	min_distance=min(agent.obs)
	if min_distance<sensor_range:
		print(min_distance)
	k=(sensor_range-min_distance)/sensor_range
	return Action( k*cos(turn_angle), k*sin(turn_angle))

def vKeepFormation(agent:Agent, xLeader, yLeader,  w)->Action:
	xDir=xLeader+agent.dx-agent.x
	yDir = yLeader + agent.dy - agent.y
	distance=sqrt(xDir**2+yDir**2)
	if distance>=w:
		return Action(w*xDir/distance, w*yDir/distance)
	else:
		return Action( xDir, yDir)
def vGoal(xLeader, yLeader, xGoal, yGoal, w)->Action:
	xDir=xGoal-xLeader
	yDir=yGoal-yLeader
	distance=sqrt(xDir**2+yDir**2)
	if distance>=w:
		return Action( w*xDir/distance, w*yDir/distance)
	else:
		return Action( xDir, yDir)