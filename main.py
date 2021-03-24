import gym
import os
import time
import torch as th
import profile

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

env_id = "gym_soccerbot:walk-forward-v0"


def train(output):
    tic = time.perf_counter()
    vec_env = make_vec_env(env_id, n_envs=12)
    # vec_env = gym.make(env_id, renders=False)
    policy_kwargs_ppo = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])])

    policy_kwargs_a2c = dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5, alpha=0.99, weight_decay=0))

    model = PPO("MlpPolicy", vec_env, n_steps=4096, batch_size=256, verbose=1, policy_kwargs=policy_kwargs_ppo,
                gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.0)
    # model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs_a2c)
    model.learn(total_timesteps=200000)
    model.save(output)

    del model
    del vec_env
    toc = time.perf_counter()
    print(f"Trained the model in {toc - tic:0.4f} seconds")


def see(output, algo):
    # model = PPO.load(output)
    model = algo.load(output)
    env = gym.make(env_id, renders=True)
    obs = env.reset()

    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.004)
        if done:
            obs = env.reset()
            print("DONE!!!!!")
            break
    env.close()


if __name__ == "__main__":
    #th.set_num_threads(1)
    #th.set_num_interop_threads(12)

    #name = "ppo_walk_test_lots"
    name = "ppo_walk_test_prof"
    # name = "a2c_walk_test"
    # train(name)
    see(name, PPO)

