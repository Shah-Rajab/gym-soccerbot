from gym.envs.registration import register

register(
    id='walk-forward-v0',
    entry_point='gym_soccerbot.envs:WalkingForward',
)

register(
    id='walk-forward-norm-v0',
    entry_point='gym_soccerbot.envs:WalkingForwardNorm',
)

register(
    id='walk-forward-v1',
    entry_point='gym_soccerbot.envs:WalkingForwardV1',
)

import os

def getDataPath():
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir
