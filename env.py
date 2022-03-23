from math import atan2
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
    def toStr(self):
        return ('['+str(self.x)+', '+str(self.y)+']')

class Env:
    def __init__(
            self, width=ENV_SIZE, height=ENV_SIZE, goal_x=GOAL_X, goal_y=GOAL_Y, N=ROBOT_NUMBER,
            desired_X=DX, desired_Y=DY, sensor_range=SENSOR_RANGE, leader_x=XL, leader_y=YL, robot_radius=ROBOT_RADIUS,
            sensor_detection_count=SENSOR_DETECTION_COUNT, buffer_size=MAX_T
    ):
        # pygame.init()
        self.width = width
        self.height = height

        self.xG = goal_x
        self.yG = goal_y
        self.N = N
        self.Dx = desired_X
        self.Dy = desired_Y
        self.sensor_range = sensor_range
        self.xL0 = leader_x
        self.yL0 = leader_y
        self.robot_radius = robot_radius
        self.sensor_detection_count = sensor_detection_count
        self.buffer_size = buffer_size

        self.v_history = numpy.zeros((self.buffer_size, self.N, 4, 2))
        self.pose_history = numpy.zeros((self.buffer_size, self.N + 1, 2))
        self.angle_history = numpy.zeros((self.buffer_size, self.N))
        self.detection_history = numpy.zeros((self.buffer_size, self.N, self.sensor_detection_count))
        self.dead_history = numpy.full((self.buffer_size, self.N), True)

        self.walls = []
        for wall_coord in self.wall_coords():
            from_x, from_y, to_x, to_y = wall_coord
            wall = Wall(from_x, from_y, to_x, to_y)
            self.walls.append(wall)
        self.reset()

    @staticmethod
    def external_wall_coords(width, height):
        return [(0, 0, 1, height), (0, height - 1, width, height), (width - 1, 0, width, height), (0, 0, width - 1, 1)]

    def wall_coords(self):
        return Env.external_wall_coords(self.width, self.height)

    def reset(self):
        self.is_done = False
        self.agents = []

        self.xL = self.xL0
        self.yL = self.yL0
        for i in range(self.N):
            agent = Agent(
                i,
                self.xL + self.Dx[i], self.yL + self.Dy[i], self.Dx[i], self.Dy[i], radius=self.robot_radius,
                sensor_detection_count=self.sensor_detection_count
            )
            self.agents.append(agent)
        self.t = 0
        self.is_done = False
        self.v_history[:] = 0
        self.pose_history[:] = 0
        self.angle_history[:] = 0
        self.detection_history[:] = 0
        self.dead_history[:] = True

    def episode_step_by_step(self, w1, w2, w3):
        self.reset()
        yield self.is_done
        while not self.is_done:
            self.play_step(w1, w2, w3)
            yield self.is_done

    def episode(self, w1, w2, w3, killed=[]):

        self.reset()
        for k in killed:
            self.agents[k].is_dead=True
        self.reset_leader()
        while not self.is_done:
            self.play_step(w1, w2, w3)

    def play_step(self, w1, w2, w3):
        # for t in range(FPS):
        self.observe()
        moving = False
        self.check_dead()
        self.reset_leader()
        for i in range(self.N):
            agent = self.agents[i]
            self.pose_history[self.t, agent.id, :] = [agent.x, agent.y]
            self.angle_history[self.t, agent.id] = agent.angle
            self.dead_history[self.t, agent.id] = agent.is_dead
            if agent.is_dead:
                continue
            v1 = v_avoid_obs_min_angle(agent, self.sensor_range)
            v2 = v_keep_formation(agent, self.xL, self.yL, w2)
            v3 = v_goal(self.xL, self.yL, self.xG, self.yG, w3)
            v = w1 * v1 + v2 + v3
            # print('i='+str(agent.id)+'v1='+v1.toStr()+', v2='+v2.toStr()+', v3='+v3.toStr()+',v='+v.toStr())
            agent.move(v)
            self.v_history[self.t, agent.id, :, :] = [[v1.x, v1.y], [v2.x, v2.y], [v3.x, v3.y], [v.x, v.y]]
            if v.active():
                moving = True

        self.t += 1
        if not moving or self.t == self.buffer_size:
            self.is_done = True

    def reset_leader(self):
        xl, yl = virtual_leader_position(self.agents)
        if xl is None or yl is None:
            self.is_done = True
        else:
            self.xL = xl
            self.yL = yl
            self.pose_history[self.t, self.N, :] = [self.xL, self.yL]

    def check_dead(self):
        for i in range(self.N):
            agent = self.agents[i]
            if agent.is_dead:
                continue
            for drawable in self.agents:
                if drawable != agent and not drawable.is_dead and agent.is_collide(drawable):
                    agent.is_dead = True
                    drawable.is_dead = True
                    break
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
                        if isinstance(drawable, Agent) and drawable.is_dead:
                            continue
                        # nn=agent.get_absolute_angle(angle)
                        dist = drawable.get_intersection(
                            x=agent.x, y=agent.y,
                            angle=angle
                        )
                        if dist == -1:
                            continue
                        # dist = max(dist-self.robot_radius, 0)
                        min_dist = min(dist, min_dist)
                if min_dist < self.sensor_range:
                    agent.obs[j] = min_dist
                    agent.detected = True
                else:
                    agent.obs[j] = -1
            self.detection_history[self.t, agent.id, :] = agent.obs

    def save_episode(self, file_name):
        numpy.savez(
            file_name, v=self.v_history[:self.t, :, :, :], pose=self.pose_history[:self.t, :, :],
            angle=self.angle_history[:self.t, :], detection=self.detection_history[:self.t, :, :],
            dead=self.dead_history[:self.t, :], width=self.width, height=self.height, goal_x=self.xG, goal_y=self.yG,
            wall=[(wall.from_x, wall.from_y, wall.to_x, wall.to_y) for wall in self.walls], radius=self.robot_radius,
            dx=self.Dx, dy=self.Dy, N=self.N, t=self.t,
            leader_x=self.xL, leader_y=self.yL
        )

    @staticmethod
    def load_episode_history(file_name):
        history = numpy.load(file_name)
        return history['v'], history['pose'], history['angle'], history['detection'], history['dead'], \
               history['width'], \
               history['height'], history['goal_x'], history['goal_y'], history['wall'], history['radius'], history[
                   'dx'], history['dy'], history['N'], history['t'], history['leader_x'], history['leader_y']

    def checkFormAchieved(self):
        for agent in self.agents:
            if not agent.is_dead and not (equals(agent.x - agent.dx, self.xL) and equals(agent.y - agent.dy, self.yL)):
                return False
        return True

    def checkGoalAchieved(self):
        for agent in self.agents:
            if not agent.is_dead and not (equals(agent.x - agent.dx, self.xG) and equals(agent.y - agent.dy, self.yG)):
                return False
        return True

    def addObstacle(self, from_x, from_y, to_x, to_y):
        self.walls.append(Wall(from_x, from_y, to_x, to_y))


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

    def __init__(self, id, x, y, dx, dy, angle=0, radius=ROBOT_RADIUS, sensor_detection_count=SENSOR_DETECTION_COUNT):
        self.angle = angle
        self.is_dead = False
        self.sensor_detection_count = sensor_detection_count
        self.obs = numpy.zeros(self.sensor_detection_count)
        self.detected = False
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
        normAngleMinusPiPi(offset - pi + i * 2 * pi / sensor_detection_count)
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
    if len(x) == 0:
        return None, None
    x = numpy.array(x)
    y = numpy.array(y)
    dx = numpy.array(dx)
    dy = numpy.array(dy)
    return numpy.mean(x - dx), numpy.mean(y - dy)


def v_avoid_obs_min_angle(agent: Agent, sensor_range) -> Action:
    if not agent.detected:
        return Action(0, 0)

    angles = numpy.array(agent.get_lidar_angles())
    agent.obs[agent.obs == -1] = sensor_range
    nom = 0
    denom = 0
    for i in range(len(angles)):
        nom += angles[i] * agent.obs[i]
        denom += agent.obs[i]

    turn_angle = nom / denom
    min_distance = min(agent.obs)
    k = (sensor_range - min_distance) / sensor_range
    return Action(k * cos(turn_angle), k * sin(turn_angle))
def v_avoid_obs_min_angle_not_null(agent: Agent, sensor_range) -> Action:
    if not agent.detected:
        return Action(0, 0)

    angles = numpy.array(agent.get_lidar_angles())
    detected_ranges= agent.obs[agent.obs != -1]
    detected_angles=angles[agent.obs!=-1]
    nom = 0
    denom = 0
    for i in range(len(detected_ranges)):
        nom += detected_angles[i] * detected_ranges[i]
        denom += detected_ranges[i]

    turn_angle = nom / denom
    min_distance = min(agent.obs)
    k = (sensor_range - min_distance) / sensor_range
    return Action(-k * cos(turn_angle), -k * sin(turn_angle))
def v_avoid_obs_min(agent: Agent, sensor_range) -> Action:
    if not agent.detected:
        return Action(0, 0)

    angles = numpy.array(agent.get_lidar_angles())


    turn_angle = opposite_angle( angles[numpy.argmin(agent.obs)])
    min_distance = min(agent.obs)
    k = (sensor_range - min_distance) / sensor_range
    return Action(k * cos(turn_angle), k * sin(turn_angle))
def v_avoid_obs_min_angle_avg(agent: Agent, sensor_range) -> Action:
    if not agent.detected:
        return Action(0, 0)

    angles = numpy.array(agent.get_lidar_angles())
    agent.obs[agent.obs == -1] = sensor_range
    sin_s = 0
    cos_s=0
    denom = 0
    for i in range(len(angles)):
        sin_s += sin(angles[i]) * agent.obs[i]
        cos_s += cos(angles[i]) * agent.obs[i]
        denom += agent.obs[i]

    turn_angle = atan2(sin_s, cos_s)
    min_distance = min(agent.obs)
    k = (sensor_range - min_distance) / sensor_range
    return Action(k * cos(turn_angle), k * sin(turn_angle))

def v_avoid_obs_min_angle_avg_notnul(agent: Agent, sensor_range) -> Action:
    if not agent.detected:
        return Action(0, 0)

    angles = numpy.array(agent.get_lidar_angles())
    # agent.obs[agent.obs == 0] = sensor_range
    detected_angles=angles[agent.obs>=0]
    detected_ranges=agent.obs[agent.obs>=0]

    sin_s = 0
    cos_s=0
    denom = 0
    for i in range(len(detected_ranges)):
        sin_s += sin(detected_angles[i]) * detected_ranges[i]
        cos_s += cos(detected_angles[i]) * detected_ranges[i]
        denom += detected_ranges[i]


    min_distance = min(agent.obs)
    k = (sensor_range - min_distance) / sensor_range
    return Action(-k * cos_s/denom, -k * sin_s/denom)

def v_avoid_obs_min_perp(agent: Agent, sensor_range) -> Action:
    if not agent.detected:
        return Action(0, 0)

    angles = numpy.array(agent.get_lidar_angles())
    # agent.obs[agent.obs == 0] = sensor_range
    detected_angles=angles[agent.obs>=0]
    detected_ranges=agent.obs[agent.obs>=0]

    sin_s = 0
    cos_s=0
    denom = 0
    for i in range(len(detected_ranges)):
        sin_s += sin(detected_angles[i]) * detected_ranges[i]
        cos_s += cos(detected_angles[i]) * detected_ranges[i]
        denom += detected_ranges[i]

    minangle= angles[ numpy.argmin(agent.obs)]

    min_distance = min(agent.obs)
    dang_angle= atan2(sin_s/denom, cos_s/denom)

    turn_angle= normAngleMinusPiPi((minangle+pi*0.5) if dang_angle<minangle else (minangle-pi*0.5))
    k = (sensor_range - min_distance) / sensor_range
    return Action(k * cos(turn_angle), k * sin(turn_angle))

def v_keep_formation(agent: Agent, leader_x, leader_y, w) -> Action:
    if w == 0:
        return Action(0.0, 0.0)
    x_dir = leader_x + agent.dx - agent.x
    y_dir = leader_y + agent.dy - agent.y
    distance = sqrt(x_dir ** 2 + y_dir ** 2)
    if distance > w:
        return Action(w * x_dir / distance, w * y_dir / distance)
    else:
        return Action(x_dir, y_dir)


def v_goal(leader_x, leader_y, goal_x, goal_y, w) -> Action:
    if w == 0:
        return Action(0.0, 0.0)
    x_dir = goal_x - leader_x
    y_dir = goal_y - leader_y
    distance = sqrt(x_dir ** 2 + y_dir ** 2)
    if distance > w:
        return Action(w * x_dir / distance, w * y_dir / distance)
    else:
        return Action(x_dir, y_dir)
