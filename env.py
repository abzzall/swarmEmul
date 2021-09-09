from typing import List

from env import Agent
from geometry import *
import numpy
from numpy import pi

ROBOT_RADIUS = 0.25
SENSOR_RANGE = 3
SENSOR_DETECTION_COUNT = 12

XL = 50
YL = 15

VIZ_SIZE = 200
ROBOT_NUMBER = 9
DX = [-SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0,
      SENSOR_RANGE + ROBOT_RADIUS, -ROBOT_RADIUS - SENSOR_RANGE, 0, ROBOT_RADIUS + SENSOR_RANGE]
DY = [-SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, 0, 0,
      SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS]

GOAL_X = 50
GOAL_Y = 90

OBSTACLE_POS = (40, 40, 60, 60)

MAX_T=3000

class Action:

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def v(self):
		return sqrt(self.x ** 2 + self.y ** 2)

	def active(self):
		return self.v() > 0.001

	def __add__(self, other):
		return Action(self.x + other.x, self.y + other.y)

	def __rmul__(self, other):
		return Action(self.x * other, self.y * other)

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ')'

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y


class Env:
	def __init__(
			self, width=VIZ_SIZE, height=VIZ_SIZE, goal_x=GOAL_X, goal_y=GOAL_Y, N=ROBOT_NUMBER,
			obstacle_pos=OBSTACLE_POS,
			desired_X=DX, desired_Y=DY, sensor_range=SENSOR_RANGE, leader_x=XL, leader_y=YL, robot_radius=ROBOT_RADIUS,
			sensor_detection_count=SENSOR_DETECTION_COUNT, buffer_size=MAX_T
	):
		# pygame.init()
		self.width = width
		self.height = height

		self.xG = goal_x
		self.yG = goal_y
		self.obstacle_pos = obstacle_pos
		self.N = N
		self.Dx = desired_X
		self.Dy = desired_Y
		self.sensor_range = sensor_range
		self.xL = leader_x
		self.yL = leader_y
		self.robot_radius = robot_radius
		self.sensor_detection_count = sensor_detection_count
		self.reset()
		self.buffer_size=buffer_size

		self.v_history=numpy.zeros((self.buffer_size, self.N, 4))
		self.pose_history=numpy.zeros((self.buffer_size, self.N+1, 2))
		self.angle_history=numpy.zeros((self.buffer_size, self.N))
		self.detection_history=numpy.zeros((self.buffer_size, self.N, self.sensor_detection_count))
		self.dead_history=numpy.full((self.buffer_size, self.N), True)

	def reset(self):
		self.is_done = False
		self.agents = []
		self.walls = []
		# external walls
		wall1 = Wall(0, 0, 1, self.height)
		wall2 = Wall(0, self.height - 1, self.width, self.height)
		wall3 = Wall(self.width - 1, 0, self.width, self.height)
		wall4 = Wall(0, 0, self.width - 1, 1)

		from_x, from_y, to_x, to_y = self.obstacle_pos
		obstacle = Wall(from_x, from_y, to_x, to_y)

		self.walls.append(wall1)
		self.walls.append(wall2)
		self.walls.append(wall3)
		self.walls.append(wall4)
		self.walls.append(obstacle)

		for i in range(self.N):
			agent = Agent(
				self.xL + self.Dx[i], self.yL + self.Dy[i], self.Dx[i], self.Dy[i], radius=self.robot_radius,
				sensor_detection_count=self.sensor_detection_count
			)
			self.agents.append(agent)
		self.t = 0
		self.is_done = False
		self.v_history[:]=0
		self.pose_history[:]=0
		self.angle_history[:]=0
		self.detection_history[:]=0
		self.dead_history[:]=True

	def play_step(self, w1, w2, w3):
		# for t in range(FPS):
		self.observe()
		moving = False
		self.t += 1
		for i in range(self.N):
			agent = self.agents[i]
			if agent.is_dead:
				continue
			v1 = v_avoid_obs_min_angle(agent, self.sensor_range)
			v2 = v_keep_formation(agent, self.xL, self.yL, w2)
			v3 = v_goal(self.xL, self.yL, self.xG, self.yG, w3)
			v = w1 * v1 + v2 + v3
			agent.move(v)
			self.v_history[self.t, agent.id, :]=[v1, v2, v3, v]
			self.pose_history[self.t, agent.id, :]=[agent.x, agent.y]
			self.angle_history[self.t, agent.id]=agent.angle
			self.dead_history[self.t, agent.id]=False
			if v.active():
				moving = True

		if not moving:
			self.is_done = True

		self.check_dead()

		self.reset_leader()

	def reset_leader(self):
		self.xL, self.yL = virtual_leader_position(self.agents)
		self.pose_history[self.t, self.N, :]=[self.xL, self.yL ]
	def check_dead(self):
		for i in range(self.N):
			agent = self.agents[i]
			if agent.is_dead:
				continue

			for drawable in self.agents:
				if drawable != agent and agent.is_collide(drawable):
					agent.is_dead = True
					drawable.is_dead = True
					break
			if not agent.is_dead:
				for drawable in self.walls:
					if agent.is_collide(drawable):
						agent.is_dead = True
						break

	def observe(self):
		for i in range(self.N):
			agent = self.agents[i]
			agent.detected = False
			for j, angle in enumerate(agent.get_lidar_angles()):
				min_dist = self.sensor_range + 1
				for drawable in self.agents + self.walls:
					if drawable != agent:
						# nn=agent.get_absolute_angle(angle)
						dist = drawable.get_intersection(
							x=agent.x, y=agent.y,
							angle=angle
						)
						if dist == -1:
							continue
						dist = max(dist, 0)
						min_dist = min(dist, min_dist)
				if min_dist <= self.sensor_range:
					agent.obs[j] = min_dist
					agent.detected = True
				else:
					agent.obs[j] = 0
			self.detection_history[self.t, agent.id, :]=agent.obs

class Drawable:
	def __init__(self, x, y, length_x, length_y):
		self.x = x
		self.y = y
		self.length_x = length_x
		self.length_y = length_y
		self.update_pos()

	def update_pos(self):
		self.from_x = self.x - self.length_x / 2
		self.from_y = self.y - self.length_y / 2
		self.to_x = self.x + self.length_x / 2
		self.to_y = self.y + self.length_y / 2

	def contains_point(self, x, y):
		return self.from_x <= x <= self.to_x and self.from_y <= y <= self.to_y

	def is_collide(self, drawable):
		return self.contains_point(drawable.from_x, drawable.from_y) \
		       or self.contains_point(drawable.from_x, drawable.to_y) \
		       or self.contains_point(drawable.to_x, drawable.to_y) \
		       or self.contains_point(drawable.to_x, drawable.from_y) \
		       or drawable.contains_point(self.from_x, self.from_y) \
		       or drawable.contains_point(self.from_x, self.to_y) \
		       or drawable.contains_point(self.to_x, self.to_y) \
		       or drawable.contains_point(self.to_x, self.from_y) \
		       or (self.from_x <= drawable.from_x and self.to_x >= drawable.to_x and self.from_y >= drawable.from_y and
		           self.to_y <= drawable.to_y) \
		       or (self.from_x >= drawable.from_x and self.to_x <= drawable.to_x and self.from_y <= drawable.from_y and
		           self.to_y >= drawable.to_y)

	def is_intersect(self, x1, y1, x2, y2):
		if (x1 < self.from_x and x2 < self.from_x) or (x1 > self.to_x and y1 > self.to_x) \
				or (y1 < self.from_y and y2 < self.to_y) \
				or (y1 > self.to_y and y2 > self.to_y):
			return False
		angle1 = calculateAngle(self.from_x, self.from_y, x1, y1, x2, y2)
		angle2 = calculateAngle(self.to_x, self.from_y, x1, y1, x2, y2)
		angle3 = calculateAngle(self.from_x, self.to_y, x1, y1, x2, y2)
		angle4 = calculateAngle(self.to_x, self.to_y, x1, y1, x2, y2)

		if ((angle1 > 0 and angle2 > 0 and angle3 > 0 and angle4 > 0) or (
				angle1 < 0 and angle2 < 0 and angle3 < 0 and angle4 < 0)):
			return False
		return True

	def get_intersection(self, x, y, angle):
		if self.contains_point(x, y):
			return 0
		if angle > pi or angle < -pi:
			return self.get_intersection(x, y, normAngleMinusPiPi(angle))
		if angle == -pi or angle == pi:
			if self.to_x <= x and self.from_y <= y <= self.to_y:
				return x - self.to_x
			else:
				return -1
		elif angle == -pi / 2:
			if self.to_y < y and self.from_x < x < self.to_x:
				return y - self.to_y
			else:
				return -1
		elif angle == 0:
			if self.from_x > x and self.from_y < y < self.to_y:
				return self.from_x - x
			else:
				return -1
		elif angle == pi / 2:
			if self.from_y > y and self.from_x < x < self.to_x:
				return self.from_y - y
			else:
				return -1
		k = tan(angle)
		b = y - k * x
		if angle < -pi / 2:
			if self.to_x < x:
				y_intersection = k * self.to_x + b
				if self.from_y <= y_intersection <= self.to_y:
					return getDistance(x, y, self.to_x, y_intersection)
			if self.to_y < y:
				x_intersection = (self.to_y - b) / k
				if self.from_x <= x_intersection <= self.to_x:
					return getDistance(x, y, x_intersection, self.to_y)
		elif -pi / 2 < angle < 0:
			if self.to_y < y:
				x_intersection = (self.to_y - b) / k
				if self.from_x <= x_intersection <= self.to_x:
					return getDistance(x, y, x_intersection, self.to_y)
			if self.from_x > x:
				y_intersection = k * self.from_x + b
				if self.from_y <= y_intersection <= self.to_y:
					return getDistance(x, y, self.from_x, y_intersection)
		elif 0 < angle < pi / 2:
			if self.from_x > x:
				y_intersection = k * self.from_x + b
				if self.from_y <= y_intersection <= self.to_y:
					return getDistance(x, y, self.from_x, y_intersection)
			if self.from_y > y:
				x_intersection = (self.from_y - b) / k
				if self.from_x <= x_intersection <= self.to_x:
					return getDistance(x, y, x_intersection, self.from_y)
		elif angle > pi / 2:
			if self.to_x < x:
				y_intersection = k * self.to_x + b
				if self.from_y <= y_intersection <= self.to_y:
					return getDistance(x, y, self.to_x, y_intersection)
			if self.from_y > y:
				x_intersection = (self.from_y - b) / k
				if self.from_x <= x_intersection <= self.to_x:
					return getDistance(x, y, x_intersection, self.from_y)
		return -1

	@property
	def from_x(self):
		if self._from_x is None:
			self._from_x = self.x - self.length_x / 2
		return self._from_x

	@property
	def from_y(self):
		if self._from_y is None:
			self._from_y = self.y - self.length_x / 2
		return self._from_y

	@property
	def to_x(self):
		if self._to_x is None:
			self._to_x = self.x + self.length_x / 2
		return self._to_x

	@property
	def to_y(self):
		if self._to_y is None:
			self._to_y = self.y + self.length_x / 2
		return self._to_y

	@from_x.setter
	def from_x(self, value):
		self._from_x = value

	@from_y.setter
	def from_y(self, value):
		self._from_y = value

	@to_x.setter
	def to_x(self, value):
		self._to_x = value

	@to_y.setter
	def to_y(self, value):
		self._to_y = value


class Agent(Drawable):
	id = 0

	def __init__(self, x, y, dx, dy, angle=0, radius=ROBOT_RADIUS, sensor_detection_count=SENSOR_DETECTION_COUNT):
		self.angle = angle
		self.is_dead = False
		self.sensor_detection_count = sensor_detection_count
		self.obs = numpy.zeros(self.sensor_detection_count)
		self.detected = False
		self.dx = dx
		self.dy = dy
		self.id = Agent.id
		Agent.id += 1
		self.radius = radius
		Drawable.__init__(self, x, y, length_x=self.radius * 2, length_y=self.radius * 2)
		self.update_pos()

	def move_polar(self, step_size, angle):
		self.angle = angle
		self.x += step_size * cos(self.angle)
		self.y += step_size * sin(self.angle)
		self.update_pos()

	def move(self, action: Action):
		self.x += action.x
		self.y += action.y
		if action.active():
			self.angle = angleWithXAxis(action.x, action.y)
		self.update_pos()

	def move_to(self, newX, newY):
		if newX == self.x and newY == self.y:
			pass
		angle = segmentAngleWithXAxis(self.x, self.y, newX, newY)
		self.x = newX
		self.y = newY
		self.angle = angle
		self.update_pos()

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
		return lidar_angles(self.sensor_detection_count, self.angle)


class Wall(Drawable):
	def __init__(self, from_x, from_y, to_x, to_y):
		w = to_x - from_x
		h = to_y - from_y
		Drawable.__init__(
			self, x=(to_x + from_x) / 2, y=(to_y + from_y) / 2, length_x=w,
			length_y=h
		)


def lidar_angles(sensor_detection_count=SENSOR_DETECTION_COUNT, offset=0):
	return [
		offset-pi + i * 2 * pi / sensor_detection_count
		for i in
		range(sensor_detection_count)]


def virtual_leader_position(agents: List[Agent]):
	x = []
	y = []
	dx = []
	dy = []
	for agent in agents:
		if not agent.is_dead:
			x.append(agent.x)
			y.append(agent.y)
			dx.append(agent.dx)
			dy.append(agent.dy)
	x = numpy.array(x)
	y = numpy.array(y)
	dx = numpy.array(dx)
	dy = numpy.array(dy)
	return numpy.mean(x - dx), numpy.mean(y - dy)


def v_avoid_obs_min_angle(agent: Agent, sensor_range) -> Action:
	if not agent.detected:
		return Action(0, 0)

	angles = numpy.array(agent.get_lidar_angles())
	agent.obs[agent.obs == 0] = sensor_range
	nom = 0
	denom = 0
	for i in range(len(angles)):
		nom += angles[i] * agent.obs[i]
		denom += agent.obs[i]

	turn_angle = nom / denom
	min_distance = min(agent.obs)
	if min_distance < sensor_range:
		print(min_distance)
	k = (sensor_range - min_distance) / sensor_range
	return Action(k * cos(turn_angle), k * sin(turn_angle))


def v_keep_formation(agent: Agent, leader_x, leader_y, w) -> Action:
	x_dir = leader_x + agent.dx - agent.x
	y_dir = leader_y + agent.dy - agent.y
	distance = sqrt(x_dir ** 2 + y_dir ** 2)
	if distance >= w:
		return Action(w * x_dir / distance, w * y_dir / distance)
	else:
		return Action(x_dir, y_dir)


def v_goal(leader_x, leader_y, goal_x, goal_y, w) -> Action:
	x_dir = goal_x - leader_x
	y_dir = goal_y - leader_y
	distance = sqrt(x_dir ** 2 + y_dir ** 2)
	if distance >= w:
		return Action(w * x_dir / distance, w * y_dir / distance)
	else:
		return Action(x_dir, y_dir)
