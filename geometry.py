from math import atan, atan2
from math import cos
from math import pi
from math import sin
from math import sqrt
from math import tan

from constants import EPSILON


def getIntersectionWithX(k, b, x):
	return k * x + b


def getDistance(x1, y1, x2, y2):
	return vectorLength(x2-x1, y2-y1)

def vectorLength(x, y):
	return sqrt(x*x+y*y)

def angleWithXAxis(x, y):
	return atan2( y, x)


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
	if angle <= pi and angle > -pi:
		return angle
	elif angle < -pi:
		while angle < -pi:
			angle = angle + 2 * pi
		return angle
	else:
		while angle > pi:
			angle = angle - 2 * pi
		return angle

def equals(a, b, eps=EPSILON):
	return abs(a-b)<=eps

def opposite_angle(angle):
	return normAngleMinusPiPi(angle+pi)

def substract_angle(angle1, angle2):
	return normAngleMinusPiPi(angle1-angle2)

def abs_differ(angle1, angle2):
	return abs(substract_angle(angle1, angle2))