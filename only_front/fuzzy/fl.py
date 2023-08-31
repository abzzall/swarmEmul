import numpy as np
from numpy import pi


def triangular_membership(x, l, p, u):
	if x <= l or x >= u:
		return 0.0
	elif l < x <= p:
		return (x - l) / (p - l)
	elif p < x < u:
		return (u - x) / (u - p)
	else:
		return 0.0


def triangular_membership_right(x, m, u):
	"""
	Triangular membership function that returns 0 if l <= x <= m, and 1 if x == u.

	Parameters:
		x (float): The input value.
		l (float): The lower bound of the triangle.
		m (float): The middle point of the triangle.
		u (float): The upper bound of the triangle.

	Returns:
		float: The membership degree of 'x' in the triangular fuzzy set.
	"""
	if m <= x <= u:
		return (x - m) / (u - m)
	else:
		return 0.0


def triangular_membership_left(x, l, m):
	"""
	Opposite triangular membership function that returns 0 if m <= x <= u, and 1 if x == l.

	Parameters:
		x (float): The input value.
		l (float): The lower bound of the triangle.
		m (float): The middle point of the triangle.
		u (float): The upper bound of the triangle.

	Returns:
		float: The membership degree of 'x' in the opposite triangular fuzzy set.
	"""
	if l <= x <= m:
		return (m - x) / (m - l)
	else:
		return 0.0


def membership_lidar_far(range, sensor_range):
	return triangular_membership_right(range, sensor_range / 2, sensor_range)


def membership_lidar_middle(range, sensor_range):
	return triangular_membership(range, sensor_range / 4, sensor_range / 2, sensor_range * 3 / 4)


def membership_lidar_close(range, sensor_range):
	return triangular_membership_left(range, 0, sensor_range / 2)


def membership_ang_sharp_left(w, W):
	return triangular_membership_left(w, -pi / 2, 2 * W - pi / 2)


def membership_ang_left(w, W):
	return triangular_membership(w, W - pi / 2, -W, 0)


def membership_ang_straight(w, W):
	return triangular_membership(w, -W, 0, W)


def membership_ang_right(w, W):
	return triangular_membership(w, 0, W, pi / 2 - W)


def membership_ang_sharp_right(w, W):
	return triangular_membership_right(w, pi / 2 - 2 * W, pi / 2)


def membership_goalside_left(w):
	return triangular_membership_left(w, -pi, 0)


def membership_goalside_straight(w, W=pi / 4):
	return triangular_membership(w, -W, 0, W)


def membership_goalside_right(w):
	return triangular_membership_right(w, 0, pi)


def fuzzify_lidar(range, sensor_range):
	return membership_lidar_close(range, sensor_range), \
		membership_lidar_middle(range, sensor_range), \
		membership_lidar_far(range, sensor_range)


def fuzzify_ang(w, W):
	return membership_ang_sharp_left(w, W), \
		membership_ang_left(w, W), \
		membership_ang_straight(w, W), \
		membership_ang_right(w, W), \
		membership_ang_sharp_right(w, W)


def fuzzy_goalside(w, W):
	return membership_goalside_left(w), \
		membership_goalside_straight(w, W), \
		membership_goalside_right(w)


def centroid_defuzzification(results, w):
	weighted_sum = 0
	total_area = 0
	for output, area in results:
		if output == 'turn right':
			weighted_sum += w * area
			total_area += area
		elif output == 'turn left':
			weighted_sum += -w * area
			total_area += area
	# For 'straight', we do not update the weighted_sum as it has no angular change

	if total_area == 0:
		return 0

	return weighted_sum / total_area


def generate_fuzzy_rules_ang_change():
	res = dict()
	# CURRENT_ANGULAR_SPEED, GOAL_SIDE, LEFT_LIDAR, CENTER LIDAR, RIGHT LIDAR
	m = {
		(0, 0, 0, 0, 0): 1,
		(0, 0, 0, 0, 1): 1,
		(0, 0, 0, 0, 2): 1,
		(0, 0, 0, 1, 0): 1,
		(0, 0, 0, 1, 1): 1,
		(0, 0, 0, 1, 2): 1,
		(0, 0, 0, 2, 0): 1,
		(0, 0, 0, 2, 1): 1,
		(0, 0, 0, 2, 2): 1,
		(0, 0, 1, 0, 0): 1,
		(0, 0, 1, 0, 1): 1,
		(0, 0, 1, 0, 2): 1,
		(0, 0, 1, 1, 0): 0,
		(0, 0, 1, 1, 1): 1,
		(0, 0, 2, 0, 0): 1,
		(0, 0, 2, 0, 1): 0,
		(0, 0, 2, 0, 2): 1,
		(0, 0, 2, 1, 0): 0,
		(0, 0, 2, 1, 1): 0,
		(0, 0, 2, 1, 2): 0,
		(0, 0, 2, 2, 0): 0,
		(0, 0, 2, 2, 1): 0,
		(0, 0, 2, 2, 2): 0,
	}
	res.update(m)
	# left
	m = {
		(1, 0, 0, 0, 0): 2,
		(1, 0, 0, 0, 1): 2,
		(1, 0, 0, 0, 2): 2,
		(1, 0, 0, 1, 0): 2,
		(1, 0, 0, 1, 1): 2,
		(1, 0, 0, 1, 2): 2,
		(1, 0, 0, 2, 0): 2,
		(1, 0, 0, 2, 1): 2,
		(1, 0, 0, 2, 2): 2,
		(1, 0, 1, 0, 0): 2,
		(1, 0, 1, 0, 1): 2,
		(1, 0, 1, 0, 2): 2,
		(1, 0, 1, 1, 0): 1,
		(1, 0, 1, 1, 1): 2,
		(1, 0, 1, 1, 2): 2,
		(1, 0, 1, 2, 0): 2,
		(1, 0, 1, 2, 1): 2,
		(1, 0, 1, 2, 2): 2,
		(1, 0, 2, 0, 0): 1,
		(1, 0, 2, 0, 1): 2,
		(1, 0, 2, 0, 2): 2,
		(1, 0, 2, 1, 0): 1,
		(1, 0, 2, 1, 1): 0,
		(1, 0, 2, 1, 2): 0,
		(1, 0, 2, 2, 0): 2,
		(1, 0, 2, 2, 1): 0,
		(1, 0, 2, 2, 2): 0
	}
	res.update(m)

	# goal will be ignored
	for key, value in res.items():
		res[(key[0], 1, key[2], key[3], key[4])] = value
		res[(key[0], 2, key[2], key[3], key[4])] = value
	# left/sharp left to right/sharp right
	for key, value in res.items():
		cur_state = 3 if key[0] == 1 else 4
		if value == 0:
			new_val = 2
		elif value == 2:
			new_val = 0
		else:
			new_val = value
		res[(cur_state, key[1], key[4], key[3], key[2])] = new_val

	res[(2, 1, 2, 2, 2)] = 1
	# random
	m_rand = dict()
	for i in range(3):
		m_rand[(2, 1, i, 0, i)] = 3
		m_rand[(2, 1, i, 1, i)] = 3
	for i in range(2):
		m_rand[(2, 1, i, 2, i)] = 3

	res.update(m_rand)
	# goal_side
	for key, value in m_rand.items():
		res[(2, 0, key[2], key[3], key[4])] = 2
		res[(2, 2, key[2], key[3], key[4])] = 0

	for i in range(3):
		for j in range(i):
			for k in range(3):
				for l in range(3):
					res[(2, k, i, l, j)]=0
					res[(2, k, j, l, i)]=2


	return res
