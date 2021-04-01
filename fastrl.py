#!/usr/bin/env python
# encoding: utf-8

# https://github.com/DerwenAI/gym_example/blob/master/train.py

from gym_soccerbot.envs.walking_forward_norm_env import WalkingForwardNorm
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import time


def main ():
    tic = time.perf_counter()
    # init directory in which to save checkpoints
    chkpt_root = os.path.join(os.path.dirname(__file__), "checkpoints/")
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = os.path.join(os.path.dirname(__file__), "ray_results/")
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = WalkingForwardNorm 
    env_id = "walk-forward-norm-v0"
    # register_env(select_env, lambda config: WalkingForwardNorm())


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["env"] = WalkingForwardNorm
    config["num_gpus"] = 1
    # config["framework"] = "torch"
    config["num_workers"] = 12
    config["num_envs_per_worker"] = 1
    config["train_batch_size"] = 16384
    config["num_sgd_iter"] = 10
    config["rollout_fragment_length"] = 512
    config["sgd_minibatch_size"] = 4096
    config["rollout_fragment_length"] = 4096
    agent = ppo.PPOTrainer(config, env=select_env)
    #register_env(select_env, lambda config: WalkingForwardNorm())

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}, total steps: {}" # saved {}"
    n_iter = 10

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        #if n % 10 == 0:
            #chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                result["timesteps_total"]))
        # ,               chkpt_file                ))
    chkpt_file = agent.save(chkpt_root)
    toc = time.perf_counter()
    print(f"Trained the model in {toc - tic:0.4f} seconds")
    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(env_id, renders=True)

    state = env.reset()
    sum_reward = 0
    n_step = 10000

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()
