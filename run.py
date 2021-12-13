import sys
from getopt import getopt

from constants import OBSTACLE_POS
from env import Env
from visualiser import episode_gui

if __name__=='__main__':
    opts, args=getopt(sys.argv[1:],'',  ['w1=', 'w2=', 'w3=', 'save='])
    save=None
    w1, w2, w3=None, None, None
    for opt, arg in opts:
        if opt =='--w1':
            w1=float(arg)
        elif opt =='--w2':
            w2=float(arg)
        elif opt =='--w3':
            w3=float(arg)
        elif opt =='--save':
            save=arg

    if w1 is None or w2 is None or w3 is None:
        raise Exception('')
    env=Env()
    obs_from_x,obs_from_y,obs_to_x,obs_to_y=OBSTACLE_POS
    env.addObstacle(obs_from_x,obs_from_y,obs_to_x,obs_to_y)
    episode_gui(env, w1, w2, w3)
    if not (save is None):
        env.save_episode(save)

