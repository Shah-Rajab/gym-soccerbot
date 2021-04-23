from gym.envs.registration import register
from ray.tune.registry import register_env

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
    id='walk-forward-norm-v1',
    entry_point='gym_soccerbot.envs:WalkingForwardNormAgn',
)

register_env("walk-forward-norm-v1", lambda config: WalkingForwardNormAgn(config))

import os

def getDataPath():
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir
