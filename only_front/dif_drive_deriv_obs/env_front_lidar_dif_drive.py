import random
from math import atan2, acos
from typing import List

from geometry import *
import numpy
from numpy import pi

from only_front.dif_drive_deriv_obs.constants import *

global S
S=0

class Action:

    def __init__(self, v, w):
        self.x = v * cos(w)
        self.y = v * sin(w)
        self.v = v
        self.w = w

    def active(self):
        return self.v >= EPSILON

    def __add__(self, other):
        return Action(self.x + other.x, self.y + other.y)

    def __rmul__(self, other):
        return Action(self.x * other, self.y * other)

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def toStr(self):
        return ('[' + str(self.x) + ', ' + str(self.y) + ']')


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

        self.v_history = numpy.zeros((self.buffer_size, self.N, 4))
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
        self.turn_side = None
        self.safe_distance = 10

    def reset_direction(self):
        self.direction = segmentAngleWithXAxis(self.xL, self.yL, self.xG, self.yG)
        return self.direction

    def get_active_agents_old(self):

        # agent in center
        center_agent = self.get_active_agent()
        dist = -1
        next = center_agent.left
        while (True):

            next_dist = abs(cos(self.direction) * (self.yL - next.y) - sin(self.direction) * (self.xL - next.x))
            if (next_dist <= dist + self.robot_radius):
                break
            dist = next_dist
            next = next.left
        left = next.right

        dist = 0
        next = center_agent.right
        while (True):

            next_dist = abs(cos(self.direction) * (self.yL - next.y) - sin(self.direction) * (self.xL - next.x))
            if (next_dist <= dist + self.robot_radius):
                break
            dist = next_dist
            next = next.right
        right = next.left
        center_agent.sensor_active = True
        left.sensor_active = True
        right.sensor_active = True
        return left, center_agent, right

    def get_active_agents(self):
        X = numpy.zeros(self.N)
        Y = numpy.zeros(self.N)
        n = 0
        alive_agent_list = []
        for agent in self.agents:
            agent.sensor_active = False
            if agent.is_dead:
                continue
            X[agent.id], Y[agent.id] = swifted_axis_coord(agent.x, agent.y, self.xL, self.yL, self.direction)
            alive_agent_list.append(agent)
            n = n + 1
        if n == 0:
            return None, None, None
        elif n == 1:
            return None, alive_agent_list[0], None
        elif n == 2:
            if Y[alive_agent_list[0].id] < Y[alive_agent_list[1].id]:
                return alive_agent_list[0], None, alive_agent_list[1]
            else:
                return alive_agent_list[1], None, alive_agent_list[0]

        def left_cond(agent):
            return -int(Y[agent.id] / self.robot_radius), X[agent.id]

        def right_cond(agent):
            return int(Y[agent.id] / self.robot_radius), X[agent.id]

        def center_cond(agent):
            return int(X[agent.id] / self.robot_radius * 0.5), -abs(Y[agent.id])

        left_list = sorted(alive_agent_list, key=left_cond)
        right_list = sorted(alive_agent_list, key=right_cond)
        center_list = sorted(alive_agent_list, key=center_cond)

        center = center_list.pop()
        left = left_list.pop()
        right = right_list.pop()
        if center.id == left.id:
            left = left_list.pop()
        if center.id == right.id:
            right = right_list.pop()
        if left.id == right.id:
            right = right_list.pop()
        return left, center, right

    def get_active_agent(self):

        direction = self.direction
        min_differ = pi
        result = None
        for agent in self.agents:
            agent.sensor_active = False

            if not agent.sensored:
                continue
            diff = abs_differ(direction, agent.cover_angle)
            if min_differ > diff:
                min_differ = diff
                result = agent

        return result

    @staticmethod
    def external_wall_coords(width, height):
        return [(0, 0, 1, height), (0, height - 1, width, height), (width - 1, 0, width, height), (0, 0, width - 1, 1)]

    def wall_coords(self):
        return Env.external_wall_coords(self.width, self.height)

    def reset(self):
        random.seed(0)
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
        self.reset_direction()
        # self.turn_speed = 0
        self.turn_left = None
        self.turn_perp = None
        self.v_left = 0
        self.v_right = 0
        self.obs_old = [self.sensor_range, self.sensor_range, self.sensor_range]

    def episode_step_by_step(self, dv, vmax):
        self.reset()
        yield self.is_done
        while not self.is_done:
            self.play_step(dv, vmax)
            yield self.is_done

    def play_step(self, dv, vmax):
        global S
        S=S+1
        print(S)
        # for t in range(FPS):
        self.reset_leader()
        # if abs(self.turn_speed) < 0.0001:
        #     self.turn_speed = 0
        # if  abs(self.turn_speed)!=pi/2:
        #     self.turn_perp=None
        # else:
        #     if self.turn_perp is None:
        #         self.turn_perp=False
        #     else:
        #         self.turn_perp=True

        obs = self.observe(pi / 12)
        turn = 0
        rand_decision = False
        goal_side = substract_angle(segmentAngleWithXAxis(self.xL, self.yL, self.xG, self.yG), self.direction)
        # safe_dist = v + self.robot_radius
        # for r in obs:
        #     if r is not None and r<safe_dist:
        #         w=pi/2

        # if obs[1] is not None and ((obs[0] is None and obs[2] is None) or (obs[0] is not None and obs[2] is not None)):#obstacle in front
        #     turn=1 if bool(random.getrandbits(1)) else -1
        # elif obs[0] is not None:
        #     turn=-1
        # elif obs[2] is not None:
        #     turn=1
        d_obs = [obs[0] - self.obs_old[0] , obs[1] - self.obs_old[1], obs[2] - self.obs_old[2]]
        range_min = min(obs)
        turn_speed = velocity_angular(self.v_left, self.v_right)
        if equals(d_obs[0], d_obs[2]):

            if goal_side > turn_speed:
                turn = 1
                # self.turn_left=False
            elif goal_side < turn_speed:
                turn = -1
                # self.turn_left=True
            elif self.v_right > self.v_left:
                turn = 1
            elif self.v_right < self.v_left:
                turn = -1
            elif d_obs[1] >= 0:
                turn = 0
                # self.turn_left=None
            else:
                # rand_decision=True
                turn = 1 if bool(random.getrandbits(1)) else -1
        else:
            turn = -1 if d_obs[0] > d_obs[2] else 1
        # if d_obs[0] is None and obs[2] is None:
        #     if obs[1] is None:
        #         self.turn_left = None
        #         turn = 0
        #
        #         # if goal_side > 0:
        #         #     turn = 1 #if self.turn_speed<=0 else -1
        #         # elif goal_side < 0:
        #         #     turn = -1 #if self.turn_speed>=0 else 1
        #     else:
        #         range_min=obs[1]
        #         rand_decision = True
        #
        #
        # else:
        #
        #     if obs[0] is None:
        #         range_min = obs[2]
        #         turn = -1
        #         # if self.turn_left == False and abs(self.turn_speed)>w:
        #         #     self.turn_speed=0
        #         # else:
        #         #     turn = -1
        #         self.turn_left = True
        #         # if self.turn_speed==0 else -1
        #     elif obs[2] is None:
        #
        #         # if self.turn_left and abs(self.turn_speed)>w:
        #         #     self.turn_speed=0
        #         # else:
        #         #     turn = 1
        #         turn = 1
        #         range_min = obs[0]
        #         self.turn_left = False
        #
        #     else:
        #         if obs[0]==obs[2]:
        #             rand_decision = True
        #         else:
        #             turn=1 if obs[0]>obs[2] else -1
        #         range_min=min([obs[0], obs[2]])
        #     if obs[1] is not None and obs[1]<range_min:
        #         range_min=obs[1]
        # if rand_decision:
        #     if self.turn_left is not None:
        #         turn = -1 if self.turn_left else 1
        #     elif self.v_left==self.v_right:
        #
        #         # if goal_side != 0:
        #         #     w = min(w, abs_differ(self.direction, segmentAngleWithXAxis(self.xL, self.yL, self.xG, self.yG)))
        #         if goal_side > 0:
        #             turn = 1
        #             range_min = obs[1]
        #         elif goal_side < 0:
        #             range_min = obs[1]
        #             turn = -1
        #         else:
        #             turn = 1 if bool(random.getrandbits(1)) else -1
        #     else:
        #         turn = 1 if self.v_right>self.v_left else -1
        #     self.turn_left = (turn == -1)
        # if obs[0]<=self.safe_distance or obs[1]<=self.safe_distance or obs[2]<=self.safe_distance:
        #     turn=1 if self.turn_speed>0 else -1
        # else:
        #     turn=0

        # if obs[0] is None:
        #     obs[0]=self.sensor_range+1
        # if obs[2] is None:
        #     obs[2]=self.sensor_range+1
        #
        # if obs[0]>obs[2]:
        #     turn=1
        # elif obs[0]<obs[2]:
        #
        #     turn=-1
        # elif obs[0]==obs[1]:
        #     if goal_side==0:
        #         turn = 1 if bool(random.getrandbits(1)) else -1
        #     elif goal_side>0:
        #         turn=-1
        #     else:
        #         turn=1
        moving = True
        dist = getDistance(self.xL, self.yL, self.xG, self.yG)
        lin_speed = velocity_linear(self.v_left, self.v_right)
        if dist == 0 and lin_speed == 0:
            moving = False
        #
        # if turn == 0:
        #     # if self.turn_speed>0:
        #     #     turn=-1
        #     # elif self.turn_speed<0:
        #     #     turn=1
        #     # if self.turn_speed != 0:
        #     #     self.turn_speed = 0
        #     # else:
        #     #     if goal_side != 0:
        #     #         w = min(w, abs_differ(self.direction, segmentAngleWithXAxis(self.xL, self.yL, self.xG, self.yG)))
        #     if goal_side > 0:
        #         turn = 1
        #
        #     elif goal_side < 0:
        #         turn = -1

        # if turn==0:
        #     if 0 > goal_side > self.turn_speed:
        #         turn = 1
        #     elif 0 < goal_side < self.turn_speed:
        #         turn = -1
        #     elif goal_side > 0:
        #         turn = 1
        #     elif goal_side < 0:
        #         turn = -1

        # if turn != 0:
        #     r = random.random()
        #     if self.turn_left is None and r < EPS_switch_on_goal:
        #         # print('goal random worked ' + str(self.t))
        #         turn = -turn
        #     elif self.turn_left is not None and r < EPS_switch_on_obs:
        #         # print('obs random worked ' + str(self.t))
        #         turn = -turn

        if turn == 0:
            n = lin_speed // dv
            if dist <= n * (n + 1) * dv / 2:
                self.v_left -= dv
                self.v_right -= dv
            else:
                if self.v_left < vmax and self.v_right < vmax:
                    self.v_left += dv
                    self.v_right += dv

        else:
            if equals(turn_speed * turn, pi / 2):
                turn = 0#-turn
                speed_up = False
            elif (turn==1 and self.v_left>=vmax) or (turn==-1 and self.v_right>=vmax):
                speed_up=False
            elif (turn == 1 and self.v_left <= 0) or (turn == -1 and self.v_right <=0):
                speed_up = True
            else:
                speed_up = abs(range_min)> ROBOT_RADIUS + abs(lin_speed)+abs(min(d_obs)) +dv


            if speed_up:
                if turn == 1:
                    self.v_left += dv
                else:
                    self.v_right += dv
            else:
                if turn == 1:
                    self.v_right -= dv
                else:
                    self.v_left -= dv
        self.obs_old=obs
        # switch_dir = (self.turn_speed > 0 and turn < 0) or (self.turn_speed < 0 and turn > 0)
        # w_old=velocity_angular(self.v_left, self.v_right)
        # if w_old == pi / 2:
        #     self.turn_speed =  turn*w
        # else:
        #     self.turn_speed = normAngleMinusPiPi(self.turn_speed + turn * w)
        #
        # if self.turn_speed > pi/2:
        #     # print('turnspeed ' + str(self.turn_speed) + ' decreased  ' + str(self.t))
        #     self.turn_speed = pi/2
        #
        # elif self.turn_speed < -pi/2:
        #     # print('turnspeed ' + str(self.turn_speed) + ' decreased  ' + str(self.t))
        #     self.turn_speed = -pi/2
        #
        # if self.turn_left is None and abs(self.turn_speed) > abs(goal_side):
        #     self.turn_speed = goal_side
        # cur_turn_speed=self.turn_speed
        # if self.turn_speed!=0:
        #     r = random.random()
        #     if self.turn_left is None and r < EPS_switch_on_goal:
        #         print('goal random worked ' + str(self.t))
        #         self.turn_speed=0
        #     elif self.turn_left is not None and r < EPS_switch_on_obs:
        #         print('obs random worked ' + str(self.t))
        #         self.turn_speed=0
        # if turn==0:
        #     print('not turning  '+str(self.t))
        # if self.turn_speed ==0:
        #     print('turn speed zero  '+str(self.t))
        # if abs(self.turn_speed)==pi/2:
        #     print('turn speed max  '+str(self.t))

        # if abs(self.turn_speed) == pi / 2 and r<EPS_decrease_on_max:
        #     self.turn_speed += (w if self.turn_speed < 0 else -w)
        self.direction = normAngleMinusPiPi(self.direction + velocity_angular(self.v_left, self.v_right))
        # if goal_side>0:
        #     self.direction+=wg
        # elif goal_side<0:
        #     self.direction-=wg
        self.check_dead()
        action = Action(velocity_linear(self.v_left, self.v_right), self.direction)
        for i in range(self.N):
            agent = self.agents[i]
            self.pose_history[self.t, agent.id, :] = [agent.x, agent.y]
            self.angle_history[self.t, agent.id] = agent.angle
            self.dead_history[self.t, agent.id] = agent.is_dead
            self.v_history[self.t, agent.id, 0] = self.v_right
            self.v_history[self.t, agent.id, 1] = self.v_left
            self.v_history[self.t, agent.id, 2] = action.x
            self.v_history[self.t, agent.id, 3] = action.y
            if agent.is_dead:
                continue
            agent.move(action)
        self.t += 1
        if not moving or self.t == self.buffer_size or self.checkGoalAchieved():
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
            if agent.x < 0 or agent.y < 0 or agent.x > self.width or agent.y > self.height:
                agent.is_dead = True
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

    def observe(self, w=0):
        # if abs(self.turn_speed) == pi / 2 or self.turn_speed == 0:
        #     angles = [self.direction - w, self.direction, self.direction + w]
        # else:
        #     angle=abs(self.turn_speed)+w
        #     if angle>pi/2:
        #         angle=pi/2
        #     if self.turn_speed < 0:
        #         angles = [self.direction -angle, self.direction, self.direction + w]
        #     else:
        #         angles = [self.direction - w, self.direction, self.direction + angle]

        left, center, right = self.get_active_agents()

        result = []
        angles = [self.direction - w, self.direction, self.direction + w]
        agents = [left, center, right]
        for i in range(3):
            agent = agents[i]
            if agent is None:
                result.append(None)
                continue
            agent.sensor_active = True
            agent.detected = False
            angle = angles[i]
            min_dist = self.sensor_range+1
            # todo: agents not implemented
            for drawable in self.walls:
                dist = drawable.get_intersection(
                    x=agent.x, y=agent.y,
                    angle=angle
                )
                if dist == -1:
                    continue
                min_dist = min(min_dist, dist)
            agent.obs = min_dist
            if min_dist <= self.sensor_range:
                agent.detected = True
            else:
                agent.obs = None
                agent.detected = True
            result.append(agent.obs)
        return result

    def save_episode(self, file_name):
        numpy.savez(
            file_name,
            v=self.v_history[:self.t, :, :],
            pose=self.pose_history[:self.t, :, :],
            angle=self.angle_history[:self.t, :],
            detection=self.detection_history[:self.t, :, :],
            dead=self.dead_history[:self.t, :],
            width=self.width,
            height=self.height,
            goal_x=self.xG,
            goal_y=self.yG,
            wall=[(wall.from_x, wall.from_y, wall.to_x, wall.to_y) for wall in self.walls],
            radius=self.robot_radius,
            dx=self.Dx,
            dy=self.Dy,
            N=self.N,
            t=self.t,
            leader_x=self.xL0,
            leader_y=self.yL0,
            sensor_range=self.sensor_range,
            sensor_detection_count=self.sensor_detection_count
        )

    @staticmethod
    def load_episode_history(file_name):
        history = numpy.load(file_name)
        return history['v'], history['pose'], history['angle'], history['detection'], history['dead'], \
            history['width'], \
            history['height'], history['goal_x'], history['goal_y'], history['wall'], history['radius'], history[
            'dx'], history['dy'], history['N'], history['t'], history['leader_x'], history['leader_y'], history[
            'sensor_range'], history['sensor_detection_count']

    @staticmethod
    def load_env_history(file_name):
        v, pose, angle, detection, dead, width, height, goal_x, goal_y, wall, radius, dx, dy, N, t, leader_x, leader_y, \
            sensor_range, sensor_detection_count = Env.load_episode_history(file_name)
        result = Env(width, height, goal_x, goal_y, N, dx, dy, sensor_range, leader_x, leader_y, radius,
                     sensor_detection_count)
        result.v_history[:t, :, :, :] = v
        result.pose_history[:t, :, :, :] = pose
        result.angle_history[:t, :] = angle
        result.detection_history[:t, :, :] = detection
        result.dead_history[:t, :] = dead

        result.walls = []
        for from_x, from_y, to_x, to_y in wall:
            result.addObstacle(from_x, from_y, to_x, to_y)

        result.t = t
        return result

    def checkFormAchieved(self):
        for agent in self.agents:
            if not agent.is_dead and not (equals(agent.x - agent.dx, self.xL) and equals(agent.y - agent.dy, self.yL)):
                return False
        return True

    def checkGoalAchieved(self):
        return getDistance(self.xL, self.yL, self.xG, self.yG) < EPSILON

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

    def __init__(self, id, x, y, dx, dy, angle=0, radius=ROBOT_RADIUS, sensor_detection_count=SENSOR_DETECTION_COUNT,
                 sensored=True):

        self.right = None
        self.left = None
        self.angle = angle
        self.is_dead = False
        self.sensor_detection_count = sensor_detection_count
        self.obs = -1
        self.detected = False
        self.dx = dx
        self.dy = dy
        self.id = id
        self.radius = radius
        Drawable.__init__(self, x, y, length_x=self.radius * 2, length_y=self.radius * 2)
        self.update_pos()
        self.sensored = sensored
        self.cover_angle = angleWithXAxis(self.dx, self.dy)
        self.sensor_active = False
        # self.left = None
        # self.right=None

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
    return numpy.mean(x), numpy.mean(y)


def velocity_angular(vL: float, vR: float, R: float = WHEEL_RADIUS, d: float = L):
    return (vL - vR) * R / d


def velocity_linear(vL: float, vR: float, R: float = WHEEL_RADIUS):
    return (vL + vR) * R / 2
