import numpy
from gym_soccerbot.envs.walking_forward_env import WalkingForward


class WalkingForwardNorm(WalkingForward):
    def __init__(self, renders=False):
        super().__init__(renders)
        self.joint_limit_high = self._joint_limit_high()
        self.joint_limit_low = self._joint_limit_low()

    def step(self, action):
        action = self.unnormalize(action, self.joint_limit_low, self.joint_limit_high)
        assert(np.logical_and.reduce(np.less_equal(action, self.joint_limit_high)), "Joint action max limit exceeded")
        assert(np.logical_and.reduce(np.more_equal(action, self.joint_limit_low)), "Joint action min limit exceeded")
        observation, reward, done, info = super().step(action)
        observation = self.normalize(observation, self.observation_limit_low, self.observation_limit_high)
        reward = reward / float(1e4)
        return observation, reward, done, info

    def reset(self):
        observation = super().reset()
        observation = self.normalize(observation, self.observation_limit_low, self.observation_limit_high)
        return observation

    @staticmethod
    def normalize(actual, low_end, high_end):
        """
        Normalizes to [-1, 1]
        :param actual: to-be-normalized value
        :param low_end: s.e.
        :param high_end: s.e.
        :return: normalized value
        """
        val = actual - low_end
        val = 2 * val / (high_end - low_end)
        val = val - 1
        return val

    @staticmethod
    def unnormalize(norm, low_end, high_end):
        """
        Unnormalizes from [-1, 1]
        :param norm: to-be-unnormalized value
        :param low_end: s.e.
        :param high_end: s.e.
        :return: actual value
        """
        val = norm + 1
        val = (val / 2) * (high_end - low_end)
        val = val + low_end
        return val






