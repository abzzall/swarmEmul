import os.path
from math import ceil
from math import cos
from math import floor
from math import sin

import numpy
import pygame
from pygame import Surface
from pygame.locals import *

import constants
from env import Agent
from env import Drawable
from env import Env
from env import Wall

from constants import *
from colors import *


def scale(v, scale_koef=1.0) -> int:

    return ceil(v * scale_koef)


def blit_drawable(drawable: Drawable, image: Surface, display: Surface, scale_koef=1.0):
    rect = image.get_rect()
    rect.center = (scale(drawable.x, scale_koef), scale(drawable.y, scale_koef))
    display.blit(image, rect)


def blit_surface(surface: Surface, display: Surface, center_x, center_y, scale_koef=1.0):
    rect = surface.get_rect()
    rect.center = (scale(center_x, scale_koef), scale(center_y, scale_koef))
    display.blit(surface, rect)


def draw_wall(wall: Wall, display: Surface, scale_koef=1.0, color=BLACK):
    draw_wall__(wall.length_x, wall.length_y, wall.x, wall.y, display, scale_koef, color)


def draw_wall__(length_x, length_y, center_x, center_y, display: Surface, scale_koef=1.0, color=BLACK):

    w = scale(length_x, scale_koef)
    h = scale(length_y, scale_koef)
    image = Surface((w, h))
    image.fill(color)
    blit_surface(image, display, center_x, center_y, scale_koef)


def draw_wall_(from_x, from_y, to_x, to_y, display: Surface, scale_koef=1.0, color=BLACK):
    w = scale(to_x - from_x, scale_koef)
    h = scale(to_y - from_y, scale_koef)
    image = Surface((w, h))
    image.fill(color)
    blit_surface(image, display, (from_x + to_x) / 2, (from_y + to_y) / 2, scale_koef)


def draw_agent(agent: Agent, display: Surface, scale_koef=1.0, color=BLUE, min_size=0):
    draw_agent_(agent.radius, agent.angle, agent.x, agent.y, display, scale_koef, color, min_size)


def draw_agent_(radius, angle, center_x, center_y, display: Surface, scale_koef=1.0, color=BLUE, min_size=0):
    l = max(scale(radius * 4, scale_koef), min_size)
    image = Surface((l, l))
    image.fill(WHITE)
    image.set_alpha(100)
    center = l / 2
    pygame.draw.circle(
        image, color, (center, center),
        scale(radius, scale_koef)
    )
    pygame.draw.line(
        image, color, (center, center),
        (center * (1 + cos(angle)), center * (1 + sin(angle)))
    )
    blit_surface(image, display, center_x, center_y, scale_koef)


def env_surface(env: Env, scale_koef=1.0, min_agent_size=0, draw_way=False) -> Surface:
    display = Surface((scale(env.width, scale_koef), scale(env.height, scale_koef)))
    display.fill(WHITE)
    for agent in env.agents:
        draw_agent(agent, display, scale_koef, color=RED if agent.is_dead else BLUE, min_size=min_agent_size)

    for wall in env.walls:
        draw_wall(wall, display, scale_koef)

    blit_surface(pygame.image.load('img/goal.png'), display, env.xG, env.yG, scale_koef)
    if not (env.xL is None or env.yL is None):
        blit_surface(pygame.image.load('img/leader.png'), display, env.xL, env.yL, scale_koef)
    if draw_way:
        draw_track(env.pose_history, env.t-1, env.N, display, scale_koef=scale_koef)
    return display


def env_surface_(
        width, height, goal_x, goal_y, wall_poses, poses, angles, deads, radius, N, scale_koef=1.0, min_agent_size=0
):
    display = Surface((scale(width, scale_koef), scale(height, scale_koef)))
    display.fill(WHITE)

    for wall_pose in wall_poses:
        draw_wall_(wall_pose[0], wall_pose[1], wall_pose[2], wall_pose[3], display, scale_koef)

    for i in range(N):
        draw_agent_(
            radius, angles[i], poses[i][0], poses[i][1], display, scale_koef, RED if deads[i] else BLUE,
            min_size=min_agent_size
        )

    blit_surface(pygame.image.load('img/goal.png'), display, goal_x, goal_y, scale_koef)

    if not (poses[N, 0] == 0 and poses[N, 1] == 0):
        blit_surface(pygame.image.load('img/leader.png'), display, poses[N, 0], poses[N, 1], scale_koef)
    return display


def draw_env_(
        display: Surface, width, height, goal_x, goal_y, wall_poses, poses, angles, deads, radius, N, min_agent_size=0
):
    scale_koef = min(display.get_width() / width, display.get_height() / height)
    surface = env_surface_(
        width, height, goal_x, goal_y, wall_poses, poses, angles, deads, radius, N, scale_koef, min_agent_size
    )
    blit_surface(surface, display, display.get_width() / 2, display.get_height() / 2)


def draw_env(env: Env, display: Surface, min_agent_size=0, draw_way=False):
    scale_koef = min(display.get_width() / env.width, display.get_height() / env.height)
    surface = env_surface(env, scale_koef, min_agent_size, draw_way)
    blit_surface(surface, display, display.get_width() / 2, display.get_height() / 2)


def clear_draw_env_(
        display: Surface, width, height, goal_x, goal_y, wall_poses, poses, angles, deads, radius, N, min_agent_size=0
):
    display.fill(WHITE)
    draw_env_(display, width, height, goal_x, goal_y, wall_poses, poses, angles, deads, radius, N, min_agent_size)
    # pygame.display.update()
    pygame.display.flip()


def clear_draw_env(env: Env, display: Surface, min_agent_size=0, draw_way=False):
    display.fill(WHITE)
    draw_env(env, display, min_agent_size, draw_way)
    # pygame.display.update()

    pygame.display.flip()


def draw_track(poses, T, n, surface, scale_koef=1.0):
    for i in range(n):
        for t in range(T):
            start_pose=[scale(poses[t, i, 0], scale_koef), scale(poses[t, i, 1], scale_koef)]
            end_pose=[scale(poses[t + 1, i, 0], scale_koef), scale(poses[t + 1, i, 1], scale_koef)]
            pygame.draw.line(surface, AQUA, start_pose ,end_pose   )
    for t in range(0, T):
        start_pose=[scale(poses[t, n, 0], scale_koef), scale(poses[t, n, 1], scale_koef)]
        end_pose=[scale(poses[t + 1, n, 0], scale_koef), scale(poses[t + 1, n, 1], scale_koef)]
        pygame.draw.line(surface, PINK, start_pose, end_pose)


def episode_gui(env: Env, w1, w2, w3, window_width=WINDOW_SIZE, window_height=WINDOW_SIZE, min_agent_size=0, fps=FPS,
                draw_way=False):
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height), HWSURFACE | DOUBLEBUF | RESIZABLE)
    pygame.display.set_caption('Симулятор движения роевых роботов')

    # env.reset()
    paused = True
    clock = pygame.time.Clock()
    clear_draw_env(env, screen, min_agent_size)
    quit = False
    env.reset_leader()
    while (not env.is_done or paused):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.display.quit()
                return
            elif event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, HWSURFACE | DOUBLEBUF | RESIZABLE)
                clear_draw_env(env, screen, min_agent_size, draw_way)

            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_UP:
                    fps = 0.9 * fps
                elif event.key == K_DOWN:
                    fps = 1.1 * fps
        if not paused:
            env.play_step(w1, w2, w3)
            clear_draw_env(env, screen, min_agent_size, draw_way)

        clock.tick(fps)


# pygame.quit()


def episode_gui_(
        w1, w2, w3, env_width=ENV_SIZE, env_height=ENV_SIZE, goal_x=GOAL_X, goal_y=GOAL_Y, N=ROBOT_NUMBER,
        obstacle_pos=OBSTACLE_POS,
        desired_X=DX, desired_Y=DY, sensor_range=SENSOR_RANGE, leader_x=XL, leader_y=YL, robot_radius=ROBOT_RADIUS,
        sensor_detection_count=SENSOR_DETECTION_COUNT, buffer_size=MAX_T, window_width=WINDOW_SIZE,
        window_height=WINDOW_SIZE, min_agent_size=0, fps=FPS
):
    env = Env(
        env_width, env_height, goal_x, goal_y, N,
        obstacle_pos,
        desired_X, desired_Y, sensor_range, leader_x, leader_y, robot_radius,
        sensor_detection_count, buffer_size
    )
    episode_gui(env, window_width, window_height, w1, w2, w3, min_agent_size, fps)


def episode_replay(
        t, v_history, pose_history, angle_history, detection_history, dead_history, env_width, env_height,
        window_width,
        window_height, goal_x, goal_y, wall_poses, radius=ROBOT_RADIUS, dX=DX,
        dY=DY, N=ROBOT_NUMBER, min_agent_size=0, fps=FPS
):
    pygame.init()
    pygame.display.set_caption('Симулятор движения роевых роботов')
    screen = pygame.display.set_mode((window_width, window_height), HWSURFACE | DOUBLEBUF | RESIZABLE)

    paused = True
    clock = pygame.time.Clock()
    i = 0
    while (i < t or paused):

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.display.quit()
                return
            elif event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, HWSURFACE | DOUBLEBUF | RESIZABLE)
                clear_draw_env_(
                    screen, env_width, env_height, goal_x, goal_y, wall_poses, pose_history[i, :, :],
                    angle_history[i, :], dead_history[i, :], radius, N
                )

            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_UP:
                    fps = 0.9 * fps
                elif event.key == K_DOWN:
                    fps = 1.1 * fps

        if not paused:
            clear_draw_env_(
                screen, env_width, env_height, goal_x, goal_y, wall_poses, pose_history[i, :, :],
                angle_history[i, :], dead_history[i, :], radius, N
            )
            i += 1

        clock.tick(fps)


def episode_replay_from_file(file_name, window_width=WINDOW_SIZE, window_height=WINDOW_SIZE, fps=FPS, min_robot_size=0):
    V, poses, angles, detections, dead, env_width, env_height, goal_x, goal_y, wall, radius, dx, dy, N, \
    t, _, _ = Env.load_episode_history(
        file_name
    )
    episode_replay(
        t, V, poses, angles, detections, dead, env_width, env_height, window_width, window_height, goal_x, goal_y,
        wall,
        radius, dx, dy, N, min_agent_size=min_robot_size, fps=fps
    )


def current_episode_replay(env: Env, window_width=WINDOW_SIZE, window_height=WINDOW_SIZE, fps=FPS):
    episode_replay(env.t, env.v_history, env.pose_history, env.angle_history, env.detection_history, env.dead_history,
                   env.width, env.height, window_width, window_height, env.xG, env.yG, env.wall_coords(),
                   env.robot_radius, env.Dx, env.Dy, env.N, fps=fps)


def desiredXYSquarePattern(N: int, d=constants.SENSOR_RANGE):
    X = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N)] * N)
    Y = numpy.array([(i - (N - 1) * 0.5) * d for i in range(N) for j in range(N)])
    return X, Y


if __name__ == '__main__':


    # episode_gui_(5, 1, 1)
    N = 2
    sensor_range = 20
    ROBOT_RADIUS=1
    Dx, Dy = desiredXYSquarePattern(N, sensor_range + ROBOT_RADIUS)
    env = Env(500, 500, 250,450, N * N, Dx, Dy, SENSOR_RANGE, 250, 50, ROBOT_RADIUS, SENSOR_DETECTION_COUNT, MAX_T)
    env.addObstacle(200, 200, 300, 300)
    # env.addObstacle(60, 80, 40, 120)

    # env.addObstacle(350, 200, 400, 250)
    # env.addObstacle(100, 200, 150, 250)
    # episode_gui(env,15, 1, 1, draw_way=True)
    W=range(20)
    run=0
    dir_name=os.path.join('..', 'run')
    for w1 in W:
        print('w1='+str(w1))
        for w2 in W:
            print('w2='+str(w2))
            for w3 in W:
                print('w3=' + str(w3))
                env.episode(w1, w2, w3)
                env.save_episode(os.path.join(dir_name, str(run)+'_'+str(w1)+'_'+str(w2)))
                run=run+1



# env.save_episode('test')
# episode_replay_from_file('test.npz', WINDOW_SIZE, WINDOW_SIZE)


