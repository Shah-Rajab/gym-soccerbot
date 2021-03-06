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

register(
    id='walk-forward-v2',
    entry_point='gym_soccerbot.envs:WalkingForwardV2',
)

register(
    id='walk-forward-v3',
    entry_point='gym_soccerbot.envs:WalkingForwardV3',
)

register(
    id='walk-forward-random-v1',
    entry_point='gym_soccerbot.envs:WalkingForwardV4',
)

register(
    id='walk-forward-velocity-v1',
    entry_point='gym_soccerbot.envs:WalkingForwardV5',
)

register(
    id='walk-forward-velocity-v2',
    entry_point='gym_soccerbot.envs:WalkingForwardV6',
)

register(
    id='walk-forward-feet-v1',
    entry_point='gym_soccerbot.envs:WalkingForwardV7',
)

register(
    id='walk-forward-velocity-v3',
    entry_point='gym_soccerbot.envs:WalkingForwardV8',
)

register(
    id='walk-forward-norm-v1',
    entry_point='gym_soccerbot.envs:WalkingForwardNormAgn',
)

import os

def getDataPath():
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir
