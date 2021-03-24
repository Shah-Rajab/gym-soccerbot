from gym.envs.registration import register

register(
    id='walk-forward-v0',
    entry_point='gym_soccerbot.envs:WalkingForward',
)

import os

def getDataPath():
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir
