from math import pi
#радиус робота
ROBOT_RADIUS = 0.5
# чувствительность лидара
SENSOR_RANGE = 5
# количество лидаров в одном роботе
SENSOR_DETECTION_COUNT = 4

# начальная координата виртуального лидера
XL = 100
YL = 30

# количество роботов
ROBOT_NUMBER = 9

# координаты роботов в требуемой геометрической фигуре относительно виртуального лидера
DX = [-SENSOR_RANGE - ROBOT_RADIUS, 0, SENSOR_RANGE + ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0,
      SENSOR_RANGE + ROBOT_RADIUS, -ROBOT_RADIUS - SENSOR_RANGE, 0, ROBOT_RADIUS + SENSOR_RANGE]
DY = [-SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, -SENSOR_RANGE - ROBOT_RADIUS, 0, 0, 0,
      SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS, SENSOR_RANGE + ROBOT_RADIUS]

# координаты цели
GOAL_X = 100
GOAL_Y = 170

# расположение преграды
OBSTACLE_POS = (80, 80, 120, 120)

# максимальное количество шагов
MAX_T=60000

# размер окна в пикселях
WINDOW_SIZE=700

# размер среды
ENV_SIZE=200

# точность вычисления
EPSILON=0.00001

FPS=5

EPS_switch_on_goal=0.001
EPS_switch_on_obs= 0.0001
EPS_decrease_on_max=0.1
MAX_ANGLE=pi/12
#Inputs:
#Previous selected turn direction: not todo
#0: LEFT
#1: STRAIGHT
#2: RIGHT
#CURENT ANGULAR SPEED:fuzzify_ang
#0: SHARP LEFT
#1: LEFT
#2: STRAIGHT
#3: RIGHT
#4: SHARP RIGHT
#GOAL SIDE
#0: Left
#1: Straight
#2: Right
#LIDAR DETECTION (LEFT, STRAIGHT, RIGHT)
#0:SHARP_LEFT
#1:LEFT
#2:STRAIGHT
#3:RIGHT
#4:SHARP_RIGHT
#OUTPUT:
#ANGULAR VELOCITY CHANGE
#0: LEFT
#1: CENTER
#2: RIGHT
FUZZY_RULES_ANGULAR_SPEED_CHANGE=[
      #CURRENT_ANGULAR_SPEED, GOAL_SIDE, LEFT_LIDAR, CENTER LIDAR, RIGHT LIDAR
      [0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 2, 1],
      [0, 0, 0, 1, 0, 1],
      [0, 0, 0, 1, 1, 1],
      [0, 0, 0, 1, 2, 1],
      [0, 0, 0, 2, 0, 1],
      [0, 0, 0, 2, 1, 1],
      [0, 0, 0, 2, 2, 1],
      [0, 0, 1, 0, 0, 1],
      [0, 0, 1, 0, 1, 1],
      [0, 0, 1, 0, 2, 1],
      [0, 0, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1],
      [0, 0, 2, 0, 0, 1],
      [0, 0, 2, 0, 1, 0],
      [0, 0, 2, 0, 2, 1],
      [0, 0, 2, 1, 0, 0],
      [0, 0, 2, 1, 1, 0],
      [0, 0, 2, 1, 2, 0],
      [0, 0, 2, 2, 0, 0],
      [0, 0, 2, 2, 1, 0],
      [0, 0, 2, 2, 2, 0],
      [1, 0, 0, 0, 0, 2],
      [1, 0, 0, 0, 1, 2],
      [1, 0, 0, 0, 2, 2],
      [1, 0, 0, 1, 0, 2],
      [1, 0, 0, 1, 1, 2],
      [1, 0, 0, 1, 2, 2],
      [1, 0, 0, 2, 0, 2],
      [1, 0, 0, 2, 1, 2],
      [1, 0, 0, 2, 2, 2],
      [1, 0, 1, 0, 0, 2],
      [1, 0, 1, 0, 1, 2],
      [1, 0, 1, 0, 2, 2],
      [1, 0, 1, 1, 0, 1],
      [1, 0, 1, 1, 1, 2],
      [1, 0, 1, 1, 2, 2],
      [1, 0, 1, 2, 0, 2],
      [1, 0, 1, 2, 1, 2],
      [1, 0, 1, 2, 2, 2],
      [1, 0, 2, 0, 0, 1],
      [1, 0, 2, 0, 1, 2],
      [1, 0, 2, 0, 2, 2],
      [1, 0, 2, 1, 0, 1],
      [1, 0, 2, 1, 1, 0],
      [1, 0, 2, 1, 2, 0],
      [1, 0, 2, 2, 0, 2],
      [1, 0, 2, 2, 1, 0],
      [1, 0, 2, 2, 2, 0],
      [2, 0, 0, 0, 0, 3],
      [2, 0, 0, 0, 1, 0],
      [2, 0, 0, 0, 2, 0],
      [2, 0, 0, 1, 0, 3],
      [2, 0, 0, 1, 1, 0],
      [2, 0, 0, 1, 2, 0],
      [2, 0, 0, 2, 0, 3],
      [2, 0, 0, 2, 1, 0],
      [2, 0, 0, 2, 2, 0],
      [2, 0, 1, 0, 0, 2],
      [2, 0, 1, 0, 1, 3],
      [2, 0, 1, 0, 2, 0],
      [2, 0, 1, 1, 0, 2],
      [2, 0, 1, 1, 1, 3],
      [2, 0, 1, 1, 2, 0],
      [2, 0, 1, 2, 0, 2],
      [2, 0, 1, 2, 1, 3],
      [2, 0, 1, 2, 2, 0],
      [2, 0, 2, 0, 0, 2],
      [2, 0, 2, 0, 1, 2],
      [2, 0, 2, 0, 2, 3],
      [2, 0, 2, 1, 0, 2],
      [2, 0, 2, 1, 1, 2],
      [2, 0, 2]




]

