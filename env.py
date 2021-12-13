from typing import List

from geometry import *
import numpy
from numpy import pi
from constants import *


class Action:

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def v(self):
		return sqrt(self.x ** 2 + self.y ** 2)

	def active(self):
		return self.v() >= EPSILON

	def __add__(self, other):
		return Action(self.x + other.x, self.y + other.y)

	def __rmul__(self, other):
		return Action(self.x * other, self.y * other)

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ')'

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y


class Env:
	def __init__(			self, width=ENV_SIZE, height=ENV_SIZE, goal_x=GOAL_X, goal_y=GOAL_Y, N=ROBOT_NUMBER,			desired_X=DX, desired_Y=DY,  leader_x=10, leader_y=10, robot_radius=ROBOT_RADIUS,			 buffer_size=MAX_T	):
		self.width = width
		self.height = height

		self.xG = goal_x
		self.yG = goal_y
		self.N = N
		self.Dx = desired_X
		self.Dy = desired_Y
		self.xL0 = leader_x
		self.yL0 = leader_y
		self.robot_radius = robot_radius
		self.buffer_size = buffer_size

		self.v_history = numpy.zeros((self.buffer_size, self.N, 3, 2))
		self.pose_history = numpy.zeros((self.buffer_size, self.N + 1, 2))
		self.angle_history = numpy.zeros((self.buffer_size, self.N))
		self.dead_history = numpy.full((self.buffer_size, self.N), True)
		self.reset()

	@staticmethod
	def external_wall_coords(width, height):
		return [(0, 0, 1, height), (0, height - 1, width, height), (width - 1, 0, width, height), (0, 0, width - 1, 1)]

	def wall_coords(self):
		return Env.external_wall_coords(self.width, self.height)

	def reset(self, reset_to_line=False):
		self.tForm=0
		self.tGoal=0
		self.form_achieved=False
		self.goal_achieved=False
		self.is_done = False
		self.walls = []
		for wall_coord in self.wall_coords():
			from_x, from_y, to_x, to_y = wall_coord
			wall = Wall(from_x, from_y, to_x, to_y)
			self.walls.append(wall)
		if reset_to_line:
			self.reset_to_line_formation()
		else:
			self.reset_formation()
		self.t = 0
		self.is_done = False
		self.v_history[:] = 0
		self.pose_history[:] = 0
		self.angle_history[:] = 0
		self.dead_history[:] = True
		self.reset_leader()

	def reset_to_line_formation(self):
		self.agents = []
		x0=self.xL0-8*(self.N-1)*0.5
		y0=self.yL0
		for i in range(self.N):
			agent = Agent(
				i,
				x0+8*i, y0, self.Dx[i], self.Dy[i], radius=self.robot_radius
			)
			self.agents.append(agent)


	def reset_to_unsimmetric_line_formation(self):
		self.agents = []
		x0=self.xL0
		y0=self.yL0
		p=-1
		for i in range(self.N):
			agent = Agent(
				i,
				i*(i+1)+5, y0, self.Dx[i], self.Dy[i], radius=self.robot_radius
			)
			self.agents.append(agent)
		self.reset_leader()

	def alive_agent_count(self):
		s = 0
		for agent in self.agents:
			if not agent.is_dead:
				s += 1
		return s

	def reset_to_custom_line_formation(self, xs: List[int], y:int):
		self.agents = []
		for i in range(self.N):
			agent = Agent(
				i,
				xs[i], y, self.Dx[i], self.Dy[i], radius=self.robot_radius
			)
			self.agents.append(agent)
		self.reset_leader()

	def reset_to_custom_pos(self, xs: List[int], ys: List[int]):
		self.agents = []
		for i in range(self.N):
			agent = Agent(
				i,
				xs[i], ys[i], self.Dx[i], self.Dy[i], radius=self.robot_radius
			)
			self.agents.append(agent)
		self.reset_leader()

	def reset_formation(self):
		self.agents = []
		self.xL = self.xL0
		self.yL = self.yL0
		for i in range(self.N):
			agent = Agent(
				i,
				self.xL + self.Dx[i], self.yL + self.Dy[i], self.Dx[i], self.Dy[i], radius=self.robot_radius
			)
			self.agents.append(agent)
	def episode_step_by_step(self, w1, w2, reset_to_line=False):
		self.reset(reset_to_line=reset_to_line)
		yield self.is_done
		while not self.is_done:
			self.play_step(w1, w2)
			yield self.is_done
	def check_form_achieved(self)->bool:
		for agent in self.agents:
			if agent.is_dead:
				continue
			if not equals(agent.x, self.xL+agent.dx) or not equals(agent.y, self.yL+agent.dy) :
				return False
		return True

	def check_goal_achieved(self):
		return equals(self.xL, self.xG) and equals(self.yL, self.yG)

	def episode(self, w1, w2, reset_to_line=False, killed_agents=None):
		self.reset(reset_to_line=reset_to_line)
		if killed_agents is not None:
			for i in killed_agents:
				self.agents[i].is_dead=True
			self.reset_leader()
		while not self.is_done:
			self.play_step(w1, w2)

	def play_step(self, w1, w2):
		moving = False
		self.check_dead()
		for i in range(self.N):
			agent = self.agents[i]
			self.pose_history[self.t, agent.id, :] = [agent.x, agent.y]
			self.angle_history[self.t, agent.id] = agent.angle
			self.dead_history[self.t, agent.id] = False
			if agent.is_dead:
				continue
			v1 = v_keep_formation(agent, self.xL, self.yL, w1)
			v2 = v_goal(self.xL, self.yL, self.xG, self.yG, w2)
			v = v1 + v2
			agent.move(v)
			self.v_history[self.t, agent.id, :, :] = [[v1.x, v1.y], [v2.x, v2.y], [v.x, v.y]]
			if v.active():
				moving = True
			# else:
			# 	v1 = v_keep_formation(agent, self.xL, self.yL, w1)
			# 	v2 = v_goal(self.xL, self.yL, self.xG, self.yG, w2)
			# 	print(str(v1)+str(v2))
		self.reset_leader()
		if not self.form_achieved and self.check_form_achieved():
			self.form_achieved=True
			self.tForm=self.t
		elif not self.check_form_achieved():
			self.form_achieved=False
		if not self.goal_achieved and self.check_goal_achieved():
			self.goal_achieved=True
			self.tGoal=self.t
		# elif self.goal_achieved and not self.check_goal_achieved():
		# 	self.goal_achieved=False



		self.t += 1
		if not moving or self.t==self.buffer_size:
			self.is_done = True

	def play_step_only_formation(self, w):
		moving = False
		self.check_dead()
		for i in range(self.N):
			agent = self.agents[i]
			self.pose_history[self.t, agent.id, :] = [agent.x, agent.y]
			self.angle_history[self.t, agent.id] = agent.angle
			self.dead_history[self.t, agent.id] = False
			if agent.is_dead:
				continue
			v = v_keep_formation(agent, self.xL, self.yL, w)
			agent.move(v)
			self.v_history[self.t, agent.id, :, :] = [[v.x, v.y], [0, 0], [v.x, v.y]]
			if v.active():
				moving = True
		self.reset_leader()
		if not self.form_achieved and self.check_form_achieved():
			self.form_achieved=True
			self.tForm=self.t
		# if not self.goal_achieved and self.check_goal_achieved():
		# 	self.goal_achieved=True
		# 	self.tGoal=self.t


		self.t += 1
		if not moving or self.t==self.buffer_size:
			self.is_done = True


	def reset_leader(self):
		xl, yl = virtual_leader_position(self.agents)
		if xl is None or yl is None:
			self.is_done = True
		elif not equals(xl, self.xL) or not equals( yl, self.yL):
			self.xL = xl
			self.yL = yl
			self.pose_history[self.t, self.N, :] = [self.xL, self.yL]

	def check_dead(self):
		for i in range(self.N):
			agent = self.agents[i]
			if agent.is_dead:
				continue

			if agent.x<0 or agent.x>self.width or agent.y<0 or agent.y>self.height:
				agent.is_dead=True
				continue
			for drawable in self.agents:
				if drawable != agent and agent.is_collide(drawable):
					agent.is_dead = True
					break
			if not agent.is_dead:
				for drawable in self.walls:
					if agent.is_collide(drawable):
						agent.is_dead = True
						break

	def save_episode(self, file_name):
		numpy.savez(
			file_name, v=self.v_history[:self.t, :, :, :], pose=self.pose_history[:self.t, :, :],
			angle=self.angle_history[:self.t, :],
			dead=self.dead_history[:self.t, :], width=self.width, height=self.height, goal_x=self.xG, goal_y=self.yG,
			wall=self.wall_coords(), radius=self.robot_radius, dx=self.Dx, dy=self.Dy, N=self.N, t=self.t,
			leader_x=self.xL, leader_y=self.yL
		)

	@staticmethod
	def load_episode_history(file_name):
		history = numpy.load(file_name)
		return history['v'], history['pose'], history['angle'], history['dead'], \
		       history['width'], \
		       history['height'], history['goal_x'], history['goal_y'], history['wall'], history['radius'], history[
			       'dx'], history['dy'], history['N'], history['t'], history['leader_x'], history['leader_y']


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

	def __init__(self, id, x, y, dx, dy, angle=0, radius=ROBOT_RADIUS):
		self.angle = angle
		self.is_dead = False
		self.dx = dx
		self.dy = dy
		self.id = id
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

class Wall(Drawable):
	def __init__(self, from_x, from_y, to_x, to_y):
		w = to_x - from_x
		h = to_y - from_y
		Drawable.__init__(
			self, x=(to_x + from_x) / 2, y=(to_y + from_y) / 2, length_x=w,
			length_y=h
		)

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
	if len(x) == 0:
		return None, None
	x = numpy.array(x)
	y = numpy.array(y)
	dx = numpy.array(dx)
	dy = numpy.array(dy)
	return numpy.mean(x - dx), numpy.mean(y - dy)



def v_keep_formation(agent: Agent, leader_x, leader_y, w) -> Action:
	if w==0:
		return Action(0.0, 0.0)
	x_dir = leader_x + agent.dx - agent.x
	y_dir = leader_y + agent.dy - agent.y
	distance = sqrt(x_dir ** 2 + y_dir ** 2)
	if distance > w:
		return Action(w * x_dir / distance, w * y_dir / distance)
	else:
		return Action(x_dir, y_dir)


def v_goal(leader_x, leader_y, goal_x, goal_y, w) -> Action:
	if w==0:
		return Action(0.0, 0.0)
	x_dir = goal_x - leader_x
	y_dir = goal_y - leader_y
	distance = sqrt(x_dir ** 2 + y_dir ** 2)
	if distance > w:
		return Action(w * x_dir / distance, w * y_dir / distance)
	else:
		return Action(x_dir, y_dir)

def v_goal1(agent: Agent, goal_x, goal_y, w) -> Action:
	if w==0:
		return Action(0.0, 0.0)
	leader_x=agent.x-agent.dx
	leader_y=agent.y-agent.dy
	x_dir = goal_x - leader_x
	y_dir = goal_y - leader_y
	distance = sqrt(x_dir ** 2 + y_dir ** 2)
	if distance > w:
		return Action(w * x_dir / distance, w * y_dir / distance)
	else:
		return Action(x_dir, y_dir)
