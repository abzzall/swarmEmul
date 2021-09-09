from math import atan
from math import ceil
from math import cos
from math import floor
from math import sin
from math import sqrt
from math import tan
from typing import List

import numpy
import pygame
from numpy import pi
from pygame.locals import *


UNIT=1
ROBOT_RADIUS=UNIT/2
FPS=60
SENSOR_RANGE= 3
SENSOR_DETECTION_COUNT = 12

XL=50
YL=15


WINDOWS_SIZE=1000
VIZ_SIZE=200
MIN_RADIUS=5

LIDAR_SENSOR_ANGLES = [
		-pi+i*2*pi/SENSOR_DETECTION_COUNT
		for i in
		range(SENSOR_DETECTION_COUNT)]

ROBOT_NUMBER=9
DX=[-SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -ROBOT_RADIUS - SENSOR_RANGE, 0, ROBOT_RADIUS + SENSOR_RANGE];
DY=[-SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, 0, 0, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS];

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

GOAL_X=50
GOAL_Y=90

OBSTACLE_POS=(40, 40, 60, 60)

W1=5
W2=1
W3=1


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
	def __init__(self, width=VIZ_SIZE, height=VIZ_SIZE, xG=GOAL_X, yG=GOAL_Y, N=ROBOT_NUMBER, obstacle_pos=OBSTACLE_POS, Dx=DX, Dy=DY, sensor_range=SENSOR_RANGE, xL=XL, yL=YL, scale_koef=1.0):
		# pygame.init()
		self.width = width
		self.height = height
		self.scale_koef = scale_koef
		Drawable.scale_koef=scale_koef
		# init display
		self.display = pygame.Surface((self.get_shown_width(), self.get_shown_height()))

		# pygame.display.set_caption('main')
		self.all_sprite_group = pygame.sprite.Group()
		# self.reset()
		pygame.display.update()

		self.xG=xG
		self.yG=yG
		self.goalImg=pygame.image.load('img/goal.png')
		self.goalRect=self.goalImg.get_rect()
		self.obstacle_pos=obstacle_pos
		self.N=N
		self.Dx=Dx
		self.Dy=Dy
		self.sensor_range=sensor_range

		self.leaderImg = pygame.image.load('img/leader.png')
		self.leaderRect = self.leaderImg.get_rect()
		self.xL=xL
		self.yL=yL

	def get_shown_width(self):
		return self.width*self.scale_koef

	def get_shown_height(self):
		return self.height * self.scale_koef

	def rescale(self, new_width, new_height):
		new_scale_koef=min(new_height/self.height, new_width/self.width)
		if new_scale_koef!=self.scale_koef:
			self.set_scaled_koef(new_scale_koef)

	def set_scaled_koef(self, scale_koef):
		self.scale_koef=scale_koef

		w=floor( self.get_shown_width())
		h=floor(self.get_shown_height())
		self.display=pygame.Surface((w,h))#pygame.transform.scale(self.display, (w,h))
		self.display.get_rect().centerx=self.display.get_rect().x+w//2
		self.display.get_rect().centery=self.display.get_rect().y+h//2
		Drawable.scale_koef=scale_koef
		for drawable in self.all_sprite_group:
			drawable.rescale()
		self.reDrawAll()

	def reDrawAll(self):
		self.reset_screen()
		self.drawSwarm()
	def reset_screen(self):
		self.display.fill((255, 255, 255))
		self.goalRect.center=(self.xG*self.scale_koef, self.yG*self.scale_koef)
		self.display.blit(self.goalImg, self.goalRect)

	def reset(self):

		self.old_v=[Action(0,0)]*self.N
		self.old_v1=[Action(0,0)]*self.N
		self.old_v2=[Action(0,0)]*self.N
		self.old_v3=[Action(0,0)]*self.N

		self.reset_screen()
		self.isDone = False
		all_sprites = []
		self.agents = []
		self.walls = []
		# external walls
		wall1 = Wall(0, 0, 1, self.height)
		wall2 = Wall(0, self.height-1 , self.width, self.height)
		wall3 = Wall(self.width-1, 0, self.width, self.height)
		wall4 = Wall(0, 0, self.width-1, 1)

		from_x, from_y, to_x, to_y=self.obstacle_pos
		obstacle=Wall(from_x, from_y, to_x, to_y)

		all_sprites.append(wall1)
		all_sprites.append(wall2)
		all_sprites.append(wall3)
		all_sprites.append(wall4)
		all_sprites.append(obstacle)

		self.walls.append(wall1)
		self.walls.append(wall2)
		self.walls.append(wall3)
		self.walls.append(wall4)
		self.walls.append(obstacle)

		for i in range(self.N):
			agent=Agent(self.xL + self.Dx[i], self.yL + self.Dy[i], self.Dx[i], self.Dy[i])

			all_sprites.append(agent)
			self.agents.append(agent)
		self.all_sprite_group.empty()
		self.all_sprite_group.add(all_sprites)
		self.drawSwarm()
		self.t=0

	def drawSwarm(self):
		self.drawLeader()
		self.all_sprite_group.draw(self.display)
	def drawLeader(self):
		self.leaderRect.center=(self.xL*self.scale_koef, self.yL*self.scale_koef)
		self.display.blit(self.leaderImg, self.leaderRect)
	def print_state(self, actions):
		for i in range(self.N):
			print(
				i, ': ', self.agents[i].x, ', ', self.agents[i].y, ', ', self.agents[i].angle, ', ', self.agents[
					i].leader_id, 'Action: ', actions[i].velocity, ', ', actions[i].angle
				)

	def play_step(self, w1, w2, w3):
		# for t in range(FPS):
		self.observe()
		moving=False
		self.t+=1
		for i in range(self.N):
			agent = self.agents[i]
			if agent.isdead:
				continue
			v1=vAvoidObsMinAngle(agent, self.sensor_range, [ agent.get_absolute_angle(angle) for angle in LIDAR_SENSOR_ANGLES ])
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
		self.reset_screen()

		self.drawLeader()

		# for agent in self.agents:
		# 	self.display.blit(agent.rect, agent.x, agent.y)
		self.all_sprite_group.draw(self.display)
		pygame.display.flip()


	def episode(self, w1, w2, w3, show_sensor=False):
		self.reset()

		while not self.isDone:
			self.play_step(w1, w2, w3)
			self.observe()
			self.show_sensor()
			pygame.display.update()

		print(self.t)

	def show_sensor(self):
		for agent in self.agents:
			for i, range in enumerate(agent.obs):
				if range<self.sensor_range:
					angle=agent.get_absolute_angle(LIDAR_SENSOR_ANGLES[i])
					pygame.draw.line(self.display, BLACK, (agent.x,agent.y ), (agent.x+range*cos(angle), agent.y+range*sin(angle)))


	def resetLeader(self):
		self.xL, self.yL=virtual_leader_position(self.agents)

	def checkDead(self):
		for i in range(self.N):
			agent = self.agents[i]
			if agent.isdead:
				continue

			for drawable in self.all_sprite_group.sprites():
				if drawable != agent and agent.isCollide(drawable):
					agent.set_dead()
					if isinstance(drawable, Agent):
						drawable.set_dead()
					break



	def observe(self):
		for i in range(self.N):
			agent = self.agents[i]
			agent.detected=False
			for j,angle in enumerate(LIDAR_SENSOR_ANGLES):
				min_dist = self.sensor_range + 1
				for drawable in self.all_sprite_group.sprites():
					if drawable != agent:
						# nn=agent.get_absolute_angle(angle)
						dist = drawable.getIntersection(
							x=agent.x, y=agent.y,
							angle=agent.get_absolute_angle(angle)
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


def angleWithXAxis(x, y):
	if x > 0:
		return atan(y / x)
	elif x==0:
		if y>0:
			return pi/2
		elif y<0:
			return -pi/2
		else:
			return 0
	else:
		if y < 0:
			return pi - atan(y / x)
		else:
			return pi + atan(y / x)


def segmentAngleWithXAxis(x1, y1, x2, y2):
	if y1 == y2:
		if x2 >= x1:
			return 0
		else:
			return -pi
	return angleWithXAxis(x2 - x1, y2 - y1)


def calculateAngle(x1, y1, x2, y2, x3, y3):
	return normAngleMinusPiPi(segmentAngleWithXAxis(x2, y2, x1, y1) - segmentAngleWithXAxis(x2, y2, x3, y3))


def normAngleMinusPiPi(angle):
	if angle <= pi and angle >= -pi:
		return angle
	elif angle < -pi:
		while angle < -pi:
			angle = angle + 2 * pi
		return angle
	else:
		while angle > pi:
			angle = angle - 2 * pi
		return angle


class Drawable(pygame.sprite.Sprite):
	scale_koef=1.0
	min_size=0

	def __init__(self, x, y, lengthX=UNIT, lengthY=UNIT, shown_width=UNIT, shown_height=UNIT):
		pygame.sprite.Sprite.__init__(self)
		self.x = x
		self.y = y
		# self.rect.center = (x, y)
		self.lengthX = lengthX
		self.lengthY = lengthY
		self.shown_width=shown_width
		self.shown_height=shown_height
		self.rescale()
		# self.image = pygame.Surface((shown_width, shown_height))
		# # self.image.fill(BLACK)
		# self.rect = self.image.get_rect()
		# self.redraw()
		self.update_pos()

	def get_scaled_width(self):
		return max(floor( self.shown_width*Drawable.scale_koef), type(self).min_size)
	def get_scaled_height(self):
			return max(floor(self.shown_height*Drawable.scale_koef), type(self).min_size)
	def rescale(self):
		self.image=pygame.Surface((self.get_scaled_width(), self.get_scaled_height()))#pygame.transform.scale(self.image, (self.get_scaled_width(), self.get_scaled_height()))
		self.rect=self.image.get_rect()
		self.rect.centerx = self.x*Drawable.scale_koef
		self.rect.centery = self.y*Drawable.scale_koef
		self.redraw()


	def redraw(self):
		pass
	def update_pos(self):
		self.rect.centerx = self.x*Drawable.scale_koef
		self.rect.centery = self.y*Drawable.scale_koef
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
		# if fromX<0 or fromY<0:
		# 	raise Exception('minus coord')
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
			# if self.fromX>x or self.fromY>y:
			# 	return -1
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


def getIntersectionWithX(k, b, x):
	return k * x + b


def getDistance(x1, y1, x2, y2):
	return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Agent(Drawable):
	id=0
	min_size=10
	def __init__(self,  x, y,dx, dy,  angle=0):
		self.angle=angle
		self.isdead = False
		self.obs=numpy.zeros(SENSOR_DETECTION_COUNT)
		self.detected=False
		self.dx=dx
		self.dy=dy
		self.id=Agent.id
		Agent.id+=1

		Drawable.__init__(self, x, y, lengthX=ROBOT_RADIUS*2, lengthY=ROBOT_RADIUS*2, shown_width=ROBOT_RADIUS*4, shown_height=ROBOT_RADIUS*4)
		self.update_pos()

	def redraw(self):
		self.image.fill(WHITE)
		self.image.set_colorkey(WHITE)
		self.image.set_alpha(100)
		radius=ROBOT_RADIUS*Drawable.scale_koef
		centerx=self.get_scaled_width()/2
		centery=self.get_scaled_height()/2
		color=RED if self.isdead else BLUE
		endx=centerx * (1 + cos(self.angle))
		endy=centery * (1 + sin(self.angle))

		pygame.draw.circle(
				self.image, color, (centerx, centery),
				radius
			)
		pygame.draw.line(
			self.image, color, (centerx, centery),
			(centerx * (1 + cos(self.angle)), centery * (1 + sin(self.angle)))
			)

	def set_angle(self, angle):
		new_angle=normAngleMinusPiPi( angle)
		if new_angle!= self.angle:
			self._set_angle(new_angle)
	def _set_angle(self, angle):
		self.angle = angle
		self.redraw()
	def set_dead(self):
		self.isdead=True
		pygame.draw.circle(self.image, RED, (ROBOT_RADIUS, ROBOT_RADIUS), ROBOT_RADIUS)

	def movePolar(self, step_size, angle):
		self.set_angle(angle)
		self.x += step_size * cos(self.angle)
		self.y += step_size * sin(self.angle)
		self.update_pos()
	def move(self, action:Action):
		self.x+=action.x
		self.y+=action.y
		if action.active():
			angle=angleWithXAxis(action.x, action.y)
			if angle != self.angle:
				self.set_angle(angle)
		self.update_pos()

	def moveTo(self, newX, newY):
		if newX==self.x and newY==self.y:
			pass
		angle=segmentAngleWithXAxis(self.x, self.y, newX, newY)
		self.x=newX
		self.y=newY
		if angle!=self.angle:
			self.set_angle(angle)
		self.update_pos()




	def get_distance(self, drawable):
		return sqrt((self.x - drawable.x) ** 2 + (self.y - drawable.y) ** 2) - ROBOT_RADIUS

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
		return angle#normAngleMinusPiPi(angle + self.angle)


class Wall(Drawable):
	def __init__(self, fromX, fromY, toX, toY):
		w=toX - fromX
		h=toY - fromY
		Drawable.__init__(
			self, x=(toX + fromX) / 2, y=(toY + fromY) / 2, lengthX=w,
			lengthY=h, shown_width=w, shown_height=h
			)

	def redraw(self):
		self.image.fill(BLACK)

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

def vAvoidObsMinAngle(agent: Agent, sensor_range, lidar_angles)->Action:
	if not agent.detected:
		return Action(0, 0)

	angles=numpy.array(lidar_angles)
	agent.obs[agent.obs==0]=sensor_range
	nom=0
	denom=0
	for i in range(SENSOR_DETECTION_COUNT):
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



def perform_event(screen, visualizer):
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.display.quit()
		elif event.type == VIDEORESIZE:
			screen = pygame.display.set_mode(event.size, HWSURFACE | DOUBLEBUF | RESIZABLE)
			visualizer.rescale(event.w, event.h)

def episode(visualizer,screen, w1, w2, w3):
	visualizer.reset()
	yield 0
	while(not visualizer.isDone):
		visualizer.play_step(w1, w2, w3)
		visualizer.show_sensor()



		yield visualizer.t

def main():
	screen = pygame.display.set_mode((WINDOWS_SIZE, WINDOWS_SIZE), HWSURFACE | DOUBLEBUF | RESIZABLE)
	visualizer=Visualizer(width=VIZ_SIZE, height=VIZ_SIZE, scale_koef=WINDOWS_SIZE/VIZ_SIZE)
	ep=episode(visualizer, screen, W1, W2, W3)
	clock = pygame.time.Clock()
	fps=FPS
	paused=False

	while(True):
		# perform_event(screen, visualizer)
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.display.quit()
			elif event.type == VIDEORESIZE:
				screen = pygame.display.set_mode(event.size, HWSURFACE | DOUBLEBUF | RESIZABLE)
				visualizer.rescale(event.w, event.h)
			elif event.type==KEYDOWN:
				if event.key==K_SPACE:
					paused= not paused
				elif event.key==K_UP:
					fps=0.9*fps
				elif event.key==K_DOWN:
					fps=1.1*fps
		if not paused:
			next(ep)
		screen.fill(WHITE)
		screen.blit(visualizer.display, (screen.get_width() / 2 - visualizer.display.get_width() / 2,
		                                 screen.get_height() / 2 - visualizer.display.get_height() / 2)
		            )
		pygame.display.update()
		pygame.display.flip()
		clock.tick(FPS)



if __name__=='__main__':
	main()

