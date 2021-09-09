from math import ceil
from math import cos
from math import floor
from math import sin

import pygame
from pygame import Surface

from env import Agent
from env import Drawable
from env import Env
from env import Wall

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
AQUA = (128, 255, 255)
BROWN = (192, 128, 64)
GRAY = (128, 128, 128)
INDIGO = (64, 72, 255)
LIME = (192, 255, 16)
ORANGE = (255, 128, 40)
PINK = (192, 64, 192)
ROSE = (255, 174, 200)
TURQUOISE = (0, 168, 255)


def scale(v, scale_koef=1.0) -> int:
	return ceil(v * scale_koef)


def blit_drawable(drawable: Drawable, image: Surface, display: Surface, scale_koef=1.0):
	display.blit(image, (scale(drawable.x, scale_koef), scale(drawable.y, scale_koef)))


def draw_wall(wall: Wall, display: Surface, scale_koef=1.0, color=BLACK):
	w = scale(wall.length_x, scale_koef)
	h = scale(wall.length_y, scale_koef)
	image = Surface((w, h))
	image.fill(color)
	blit_drawable(wall, image, display, scale_koef)


def draw_agent(agent: Agent, display: Surface, scale_koef=1.0, color=BLUE, min_size=0):
	l = max(scale(agent.radius * 4, scale_koef), min_size)
	image = Surface((l, l))
	center = l / 2
	pygame.draw.circle(
		image, color, (center, center),
		agent.radius
	)
	pygame.draw.line(
		image, color, (center, center),
		(center * (1 + cos(agent.angle)), center * (1 + sin(agent.angle)))
	)
	blit_drawable(agent, image, display, scale_koef)


def env_surface(env: Env, scale_koef=1.0, min_agent_size=0) -> Surface:
	display = Surface((scale(env.width, scale_koef), scale(env.height, scale_koef)))
	display.fill(WHITE)
	for agent in env.agents:
		draw_agent(agent, display, scale_koef, color=RED if agent.is_dead else BLUE, min_size=min_agent_size)

	for wall in env.walls:
		draw_wall(wall, display, scale_koef)

	goalImg = pygame.image.load('img/goal.png')
	display.blit(goalImg, (scale(env.xG, scale_koef), scale(env.yG, scale_koef)))

	leaderImg = pygame.image.load('img/leader.png')
	display.blit(leaderImg, (scale(env.xL, scale_koef), scale(env.yL, scale_koef)))
	return display


def draw_env(env: Env, display: Surface, min_agent_size=0):
	scale_koef = min(display.get_width() / env.width, display.get_height() / env.height)
	display.blit(env_surface(env, scale_koef, min_agent_size), (display.get_width() / 2, display.get_height() / 2))

