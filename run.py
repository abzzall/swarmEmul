import sys
from getopt import getopt

from env import Env
from visualiser import episode_gui

if __name__=='__main__':
    opts, args=getopt(sys.argv[1:],'',  ['w1=', 'w2=', 'w3=', 'save='])
    save=None
    w1, w2=None, None
    for opt, arg in opts:
        if opt =='--w1':
            w1=float(arg)
        elif opt =='--w2':
            w2=float(arg)
        elif opt =='--save':
            save=arg

    if w1 is None or w2 is None:
        raise Exception('')
    env=Env()

    episode_gui(env, w1, w2)
    if not (save is None):
        env.save_episode(save)