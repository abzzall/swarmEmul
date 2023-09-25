import random

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
	if range is None:
		return 0, 0, 1
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


# ANGULAR VELOCITY CHANGE
# 0: LEFT
# 1: CENTER
# 2: RIGHT
def centroid_defuzzification(results, w):
	weighted_sum = 0
	total_area = 0
	for output, area in enumerate(results):
		if output == 2:  # 'turn right':
			weighted_sum += w * area
			total_area += area
		elif output == 0:  # 'turn left':
			weighted_sum += -w * area
			total_area += area
		elif output == 1:  # todo check if necessary
			total_area += area
		elif output == 3:  # random
			weighted_sum += area * w if bool(random.getrandbits(1)) else -area * w
			total_area += area
	if total_area == 0:
		return 0

	return weighted_sum / total_area


# CURRENT_ANGULAR_SPEED, GOAL_SIDE, LEFT_LIDAR, CENTER LIDAR, RIGHT LIDAR
# Apply fuzzy rules and compute the output
def apply_fuzzy_rules(CURRENT_ANGULAR_SPEED, GOAL_SIDE, LEFT_LIDAR, CENTER_LIDAR, RIGHT_LIDAR, fuzzy_rules):
	output_fuzzy = np.zeros(4)  # Initialize an array to hold the fuzzy output

	for rule in fuzzy_rules:
		# Rule format: [Temperature_index, Humidity_index, Light_index, CO2_index, Fan_Speed_index]
		# For each rule, we access the indices corresponding to the temperature, humidity, light intensity, CO2 level,
		# and fan speed

		# Apply the minimum operator between the fuzzy membership degrees of all input variables in the rule.
		# This represents the degree of support for this rule.
		# rule_support = np.minimum(CURRENT_ANGULAR_SPEED[rule[0]], GOAL_SIDE[rule[1]], LEFT_LIDAR[rule[2]],
		#                           CENTER_LIDAR[rule[3]], RIGHT_LIDAR[rule[4]])
		rule_support = np.minimum(CURRENT_ANGULAR_SPEED[rule[0]], GOAL_SIDE[rule[1]])
		rule_support = np.minimum(rule_support, LEFT_LIDAR[rule[2]])
		rule_support = np.minimum(rule_support, CENTER_LIDAR[rule[3]])
		rule_support = np.minimum(rule_support, RIGHT_LIDAR[rule[4]])
		# Update the fuzzy output based on the degree of support for this rule.
		# Use the maximum operator to capture the OR operation implied by the fuzzy rules.
		# If a rule provides support for a certain fan speed, it increases the membership degree of that fan speed in 
		# the output_fuzzy array.
		output_fuzzy[rule[5]] = np.maximum(output_fuzzy[rule[5]], rule_support)
	# print(output_fuzzy)
	return output_fuzzy


def generate_fuzzy_rules_ang_change():
	res = []
	# CURRENT_ANGULAR_SPEED, GOAL_SIDE, LEFT_LIDAR, CENTER LIDAR, RIGHT LIDAR
	#
	m = [
		[0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 1, 1],
		[0, 0, 0, 0, 2, 1],
		[0, 0, 0, 1, 0, 1],
		[0, 0, 0, 1, 1, 2],
		[0, 0, 0, 1, 2, 2],
		[0, 0, 0, 2, 0, 1],
		[0, 0, 0, 2, 1, 2],
		[0, 0, 0, 2, 2, 2],
		[0, 0, 1, 0, 0, 2],
		[0, 0, 1, 0, 1, 2],
		[0, 0, 1, 0, 2, 2],
		[0, 0, 1, 1, 0, 2],
		[0, 0, 1, 1, 1, 2],
		[0, 0, 2, 0, 0, 2],
		[0, 0, 2, 0, 1, 2],
		[0, 0, 2, 0, 2, 2],
		[0, 0, 2, 1, 0, 2],
		[0, 0, 2, 1, 1, 2],
		[0, 0, 2, 1, 2, 2],
		[0, 0, 2, 2, 0, 2],
		[0, 0, 2, 2, 1, 2],
		[0, 0, 2, 2, 2, 2],
	]
	res += m
	# left
	m = [
		[1, 0, 0, 0, 0, 0],
		[1, 0, 0, 0, 1, 0],
		[1, 0, 0, 0, 2, 0],
		[1, 0, 0, 1, 0, 0],
		[1, 0, 0, 1, 1, 0],
		[1, 0, 0, 1, 2, 0],
		[1, 0, 0, 2, 0, 0],
		[1, 0, 0, 2, 1, 0],
		[1, 0, 0, 2, 2, 0],
		[1, 0, 1, 0, 0, 0],
		[1, 0, 1, 0, 1, 1],
		[1, 0, 1, 0, 2, 1],
		[1, 0, 1, 1, 0, 1],
		[1, 0, 1, 1, 1, 2],
		[1, 0, 1, 1, 2, 2],
		[1, 0, 1, 2, 0, 2],
		[1, 0, 1, 2, 1, 2],
		[1, 0, 1, 2, 2, 2],
		[1, 0, 2, 0, 0, 1],
		[1, 0, 2, 0, 1, 1],
		[1, 0, 2, 0, 2, 1],
		[1, 0, 2, 1, 0, 2],
		[1, 0, 2, 1, 1, 2],
		[1, 0, 2, 1, 2, 2],
		[1, 0, 2, 2, 0, 2],
		[1, 0, 2, 2, 1, 2],
		[1, 0, 2, 2, 2, 2]
	]
	res += m
	m = []
	# goal will be ignored
	for row in res:
		m.append([row[0], 1, row[2], row[3], row[4], row[5]])
		m.append([row[0], 2, row[2], row[3], row[4], row[5]])
	res += m
	# left/sharp left to right/sharp right
	m = []
	for row in res:
		cur_state = 3 if row[0] == 1 else 4
		value = row[5]
		if value == 0:
			new_val = 2
		elif value == 2:
			new_val = 0
		else:
			new_val = value
		m.append([cur_state, row[1], row[4], row[3], row[2], new_val])
	res += m
	res.append([2, 1, 2, 2, 2, 1])
	# random
	m_rand = []
	for i in range(3):
		m_rand.append([2, 1, i, 0, i, 3])
		m_rand.append([2, 1, i, 1, i, 3])
	for i in range(2):
		m_rand.append([2, 1, i, 2, i, 3])

	res += m_rand
	m = []
	# goal_side
	for row in m_rand:
		m.append([2, 0, row[2], row[3], row[4], 0])
		m.append([2, 2, row[2], row[3], row[4], 2])
	res += m
	for i in range(3):
		for j in range(i):
			for k in range(3):
				for l in range(3):
					res.append([2, k, i, l, j, 0])#2, y, 1, x, 0, 0
					res.append([2, k, j, l, i, 2])

	return res


fuzzy_rules = generate_fuzzy_rules_ang_change()


def get_velocity(current_angular_speed, goal_side, left_lidar, center_lidar, right_lidar, W, sensor_range):
	print(current_angular_speed, goal_side, left_lidar, center_lidar, right_lidar, W, sensor_range)
	current_angular_speed_ = fuzzify_ang(current_angular_speed, W)
	goal_side_ = fuzzy_goalside(goal_side, W)
	left_lidar_ = fuzzify_lidar(left_lidar, sensor_range)
	center_lidar_ = fuzzify_lidar(center_lidar, sensor_range)
	right_lidar_ = fuzzify_lidar(right_lidar, sensor_range)
	print(
		current_angular_speed_,
		goal_side_,
		left_lidar_,
		center_lidar_,
		right_lidar_
		)
	fuzzy_result = apply_fuzzy_rules(
		current_angular_speed_,
		goal_side_,
		left_lidar_,
		center_lidar_,
		right_lidar_,
		fuzzy_rules
	)
	print(fuzzy_result)
	return centroid_defuzzification(
		fuzzy_result,
		W
	)
