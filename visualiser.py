from math import ceil
from math import cos
from math import floor
from math import sin

import pygame
from pygame import Surface
from pygame.locals import *

from env import Agent
from env import Drawable
from env import Env
from env import Wall

from constants import *
def scale(v, scale_koef=1.0) -> int:
	return ceil(v * scale_koef)


def blit_drawable(drawable: Drawable, image: Surface, display: Surface, scale_koef=1.0):
	rect=image.get_rect()
	rect.center=(scale(drawable.x, scale_koef), scale(drawable.y, scale_koef))
	display.blit(image, rect)


def draw_wall(wall: Wall, display: Surface, scale_koef=1.0, color=BLACK):
	w = scale(wall.length_x, scale_koef)
	h = scale(wall.length_y, scale_koef)
	image = Surface((w, h))
	image.fill(color)
	blit_drawable(wall, image, display, scale_koef)


def draw_agent(agent: Agent, display: Surface, scale_koef=1.0, color=BLUE, min_size=0):
	l = max(scale(agent.radius * 4, scale_koef), min_size)
	image = Surface((l, l))
	image.fill(WHITE)
	image.set_alpha(100)
	center = l / 2
	pygame.draw.circle(
		image, color, (center, center),
		scale(agent.radius, scale_koef)
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
	goal_rect=goalImg.get_rect()
	goal_rect.center=(scale(env.xG, scale_koef), scale(env.yG, scale_koef))
	display.blit(goalImg, goal_rect)
	if not(env.xL is None or env.yL is None):
		leaderImg = pygame.image.load('img/leader.png')
		leader_rect=leaderImg.get_rect()
		leader_rect.center=(scale(env.xL, scale_koef), scale(env.yL, scale_koef))
		display.blit(leaderImg, leader_rect)
	return display


def draw_env(env: Env, display: Surface, min_agent_size=0):
	scale_koef = min(display.get_width() / env.width, display.get_height() / env.height)
	surface=env_surface(env, scale_koef, min_agent_size)
	rect=surface.get_rect()
	rect.center=(display.get_width() / 2, display.get_height() / 2)
	display.blit(surface, rect)

def clear_draw_env(env: Env, display:Surface, min_agent_size=0):
	display.fill(WHITE)
	draw_env(env, display, min_agent_size)
	pygame.display.update()
	pygame.display.flip()


def episode_gui(env: Env, window_width, window_height, w1, w2, w3, min_agent_size=0, fps=FPS):
	pygame.init()
	screen = pygame.display.set_mode((window_width, window_height), HWSURFACE | DOUBLEBUF | RESIZABLE)

	env.reset()
	paused=False
	clock=pygame.time.Clock()
	clear_draw_env(env, screen, min_agent_size)

	while(not env.is_done or paused):
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.display.quit()
			elif event.type == VIDEORESIZE:
				screen = pygame.display.set_mode(event.size, HWSURFACE | DOUBLEBUF | RESIZABLE)
				clear_draw_env(env,screen, min_agent_size)

			elif event.type==KEYDOWN:
				if event.key==K_SPACE:
					paused= not paused
				elif event.key==K_UP:
					fps=0.9*fps
				elif event.key==K_DOWN:
					fps=1.1*fps
			if not paused:
				env.play_step(w1, w2, w3)
				clear_draw_env(env, screen, min_agent_size)

		clock.tick(fps)
	pygame.quit()


def episode_gui_(w1, w2, w3,env_width=ENV_SIZE, env_height=ENV_SIZE, goal_x=GOAL_X, goal_y=GOAL_Y, N=ROBOT_NUMBER,
			obstacle_pos=OBSTACLE_POS,
			desired_X=DX, desired_Y=DY, sensor_range=SENSOR_RANGE, leader_x=XL, leader_y=YL, robot_radius=ROBOT_RADIUS,
			sensor_detection_count=SENSOR_DETECTION_COUNT, buffer_size=MAX_T, window_width=WINDOW_SIZE, window_height=WINDOW_SIZE,  min_agent_size=0, fps=FPS):
	env=Env(env_width, env_height, goal_x, goal_y, N,
			obstacle_pos,
			desired_X, desired_Y, sensor_range, leader_x, leader_y, robot_radius,
			sensor_detection_count, buffer_size)
	episode_gui(env,window_width, window_height, w1, w2, w3, min_agent_size, fps)

if __name__=='__main__':
	episode_gui_(5, 1, 1)