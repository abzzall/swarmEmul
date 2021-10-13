from math import atan
from math import cos
from math import pi
from math import sin
from math import sqrt
from math import tan

from constants import EPSILON


def getIntersectionWithX(k, b, x):
	return k * x + b

def equals(a, b, h=EPSILON):
	return abs(a-b)<h

def getDistance(x1, y1, x2, y2):
	return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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
