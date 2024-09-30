from math import pi
#радиус робота
ROBOT_RADIUS = 1.5

WHEEL_RADIUS = 0.5
L=2

# чувствительность лидара
SENSOR_RANGE = 5
# количество лидаров в одном роботе
SENSOR_DETECTION_COUNT = 12

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


