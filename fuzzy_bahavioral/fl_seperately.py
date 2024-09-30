import random

import numpy as np
from numpy import pi

from fuzzy_bahavioral.constants import echo


def triangular_membership(x, l, p, u):
	if x <= l or x >= u:
		return 0.0
	elif l < x <= p:
		return (x - l) / (p - l)
	elif p < x < u:
		return (u - x) / (u - p)
	else:
		return 0.0


def triangular_membership_right(x, m, u, restricted=False):
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
	elif x>u and not restricted:
		return 1.0
	else:
		return 0.0


def triangular_membership_left(x, l, m, restricted=False):
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
	elif x<=l and not restricted:
		return 1.0
	else:
		return 0.0

def membership_lidar_far(range, sensor_range):
	if range is None:
		return 0#1
	else:
		return triangular_membership_right(range, 0, sensor_range, restricted=True)


def membership_lidar_close(range, sensor_range):
	if range is None:
		return 0
	return triangular_membership_left(range, 0, sensor_range)
def membership_lidar_not_detected(range, sensor_range):
	if range is None or range>sensor_range:
		return 1
	else:
		return 0

def membership_wall_following_on(x):
	if x:
		return 1
	else:
		return 0

def membership_wall_following_off(x):
	if not x:
		return 1
	else:
		return 0


def membership_des_dist_far(range, V):
	if range >V:
		return 1.0
	return triangular_membership_right(range, 0, V)


def membership_des_dist_close(range, V):
	if range >V:
		return 0.0
	return triangular_membership_left(range, 0, V)


def fuzzify_lidar(range, sensor_range):
	return membership_lidar_close(range, sensor_range), \
		membership_lidar_far(range, sensor_range),\
		membership_lidar_not_detected(range, sensor_range)

def fuzzify_des_dist(des_dist, V):
	return membership_des_dist_close(des_dist, V), \
		membership_des_dist_far(des_dist, V)

def fuzzify_wall_follow(x):
	return membership_wall_following_off(x),\
			membership_wall_following_on(x)

#0: W1,
#1: W2
#2: W3
def centroid_defuzzification(results):
	coefs=np.zeros(4)
	for i, r in enumerate(results):
		weighted_sum = 0
		total_area = 0
		for output, area in enumerate(r):
			if output == 0:  # 'turn right':
				weighted_sum += 0 * area
				total_area += area
			elif output == 1:  # 'turn left':
				weighted_sum += 0.5 * area
				total_area += area
			elif output == 2:  # todo check if necessary
				weighted_sum += 1 * area
				total_area += area
		if total_area == 0:
			coefs[i]=0
		else:
			coefs[i]=weighted_sum / total_area
	# print(f'coefs = {coefs}, normalised coefs = {coefs/np.sum(coefs)}')
	return (coefs/np.sum(coefs))
# CURRENT_ANGULAR_SPEED, GOAL_SIDE, LEFT_LIDAR, CENTER LIDAR, RIGHT LIDAR
# Apply fuzzy rules and compute the output
def apply_fuzzy_rules(LIDAR_RANGE, DES_DIST, fuzzy_rules):
	output_fuzzy = np.zeros((3,3))  # Initialize an array to hold the fuzzy output

	for rule in fuzzy_rules:
		# Rule format: [Temperature_index, Humidity_index, Light_index, CO2_index, Fan_Speed_index]
		# For each rule, we access the indices corresponding to the temperature, humidity, light intensity, CO2 level,
		# and fan speed

		# Apply the minimum operator between the fuzzy membership degrees of all input variables in the rule.
		# This represents the degree of support for this rule.
		# rule_support = np.minimum(CURRENT_ANGULAR_SPEED[rule[0]], GOAL_SIDE[rule[1]], LEFT_LIDAR[rule[2]],
		#                           CENTER_LIDAR[rule[3]], RIGHT_LIDAR[rule[4]])
		rule_support = np.minimum(LIDAR_RANGE[rule[0]], DES_DIST[rule[1]])
		# Update the fuzzy output based on the degree of support for this rule.
		# Use the maximum operator to capture the OR operation implied by the fuzzy rules.
		# If a rule provides support for a certain fan speed, it increases the membership degree of that fan speed in
		# the output_fuzzy array.
		output_fuzzy[0][rule[2]] = np.maximum(output_fuzzy[0][rule[2]], rule_support)
		output_fuzzy[1][rule[3]] = np.maximum(output_fuzzy[1][rule[3]], rule_support)
		output_fuzzy[2][rule[4]] = np.maximum(output_fuzzy[2][rule[4]], rule_support)

	# print(f'output_fuzzy: {output_fuzzy}, sum: {sum(output_fuzzy)}')
	return output_fuzzy

def apply_fuzzy_rules1(LIDAR_RANGE, DES_DIST, WALL_FOLLOW, fuzzy_rules):
	output_fuzzy = np.zeros((4, 3))  # Initialize an array to hold the fuzzy output

	for rule in fuzzy_rules:
		# Rule format: [Temperature_index, Humidity_index, Light_index, CO2_index, Fan_Speed_index]
		# For each rule, we access the indices corresponding to the temperature, humidity, light intensity, CO2 level,
		# and fan speed

		# Apply the minimum operator between the fuzzy membership degrees of all input variables in the rule.
		# This represents the degree of support for this rule.
		# rule_support = np.minimum(CURRENT_ANGULAR_SPEED[rule[0]], GOAL_SIDE[rule[1]], LEFT_LIDAR[rule[2]],
		#                           CENTER_LIDAR[rule[3]], RIGHT_LIDAR[rule[4]])
		rule_support = np.minimum(LIDAR_RANGE[rule[0]], np.minimum(DES_DIST[rule[1]], WALL_FOLLOW[rule[2]]))
		# Update the fuzzy output based on the degree of support for this rule.
		# Use the maximum operator to capture the OR operation implied by the fuzzy rules.
		# If a rule provides support for a certain fan speed, it increases the membership degree of that fan speed in
		# the output_fuzzy array.
		output_fuzzy[0][rule[3]] = np.maximum(output_fuzzy[0][rule[3]], rule_support)
		output_fuzzy[1][rule[4]] = np.maximum(output_fuzzy[1][rule[4]], rule_support)
		output_fuzzy[2][rule[5]] = np.maximum(output_fuzzy[2][rule[5]], rule_support)
		output_fuzzy[3][rule[6]] = np.maximum(output_fuzzy[3][rule[6]], rule_support)

	# print(f'output_fuzzy: {output_fuzzy}, sum: {sum(output_fuzzy)}')
	return output_fuzzy

def generate_fuzzy_rules_ang_change():
	return [
		[0, 0, 2, 0, 0],
		[0, 1, 2, 2, 0],
		[1, 0, 2, 0, 0],
		[1, 1, 2, 0, 0],
		[2, 0, 0, 0, 2],
		[2, 1, 0, 2, 0]
	]

#input: lidar range, des_dis, wall
#output: w1-4
def generate_fuzzy_rules_ang_change1():
	return [
		[0, 0, 0, 2, 0, 0, 0],
		[0, 0, 1, 2, 0, 0, 2],

		[0, 1, 0, 2, 0, 0, 0],
		[0, 1, 1, 2, 0, 0, 2],

		[1, 0, 0, 2, 0, 2, 0],
		[1, 0, 1, 2, 0, 0, 2],

		[1, 1, 0, 2, 0, 2, 0],
		[1, 1, 1, 2, 0, 0, 2],

		[2, 0, 0, 0, 0, 2, 0],
		[2, 0, 1, 0, 0, 0, 2],

		[2, 1, 0, 0, 2, 2, 0],
		[2, 1, 1, 0, 0, 0, 2]
	]


def generate_fuzzy_rules_ang_change2():
	return [
		[0, 0, 0, 2, 0, 0],
		[0, 0, 1, 2, 0, 0],

		[0, 1, 0, 2, 2, 0],
		[0, 1, 1, 2, 0, 0],

		[1, 0, 0, 2, 0, 0],
		[1, 0, 1, 2, 0, 0],

		[1, 1, 0, 2, 2, 0],
		[1, 1, 1, 2, 0, 0],

		[2, 0, 0, 0, 0, 2],
		[2, 0, 1, 2, 0, 2],

		[2, 1, 0, 0, 2, 0],
		[2, 1, 1, 2, 2, 0]
	]

fuzzy_rules = generate_fuzzy_rules_ang_change1()


def get_koefs(lidar_range, des_dist, V, sensor_range):
	# print(current_angular_speed, goal_side, left_lidar, center_lidar, right_lidar, W, sensor_range)
	lidar_range_ = fuzzify_lidar(lidar_range, sensor_range)
	des_dist_ = fuzzify_des_dist(des_dist, V)
	print(f'lidar_range={lidar_range}, fuzzy:{lidar_range_}')
	print(f'des_dist={des_dist}, fuzzy:{des_dist_}')
	# print(
	# 	current_angular_speed_,
	# 	goal_side_,
	# 	left_lidar_,
	# 	center_lidar_,
	# 	right_lidar_
	# 	)
	fuzzy_result = apply_fuzzy_rules(
		lidar_range_, des_dist_,
		fuzzy_rules
	)
	result=centroid_defuzzification(
		fuzzy_result)
	print(f'fuzzy_result: {fuzzy_result}, result: {result}, sum: {sum(result)}')
	return result

def get_koefs1(lidar_range, des_dist, wall_following, V, sensor_range, id=0):
	# print(current_angular_speed, goal_side, left_lidar, center_lidar, right_lidar, W, sensor_range)
	lidar_range_ = fuzzify_lidar(lidar_range, sensor_range)
	des_dist_ = fuzzify_des_dist(des_dist, V)
	wall_following_=fuzzify_wall_follow(wall_following)
	echo(id,f'lidar_range={lidar_range}, fuzzy:{lidar_range_}')
	echo(id,f'des_dist={des_dist}, fuzzy:{des_dist_}')
	echo(id,f'wall_follow={wall_following}, fuzzy:{wall_following_}')
	# print(
	# 	current_angular_speed_,
	# 	goal_side_,
	# 	left_lidar_,
	# 	center_lidar_,
	# 	right_lidar_
	# 	)
	fuzzy_result = apply_fuzzy_rules1(
		lidar_range_, des_dist_, wall_following_,
		fuzzy_rules
	)
	result=centroid_defuzzification(
		fuzzy_result)
	echo(id, f'fuzzy_result: {fuzzy_result}, result: {result}, sum: {sum(result)}')
	return result