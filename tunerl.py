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
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

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


#if __name__ == "__main__":
#    main()
class TestKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(TestKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_2 = tf.keras.layers.Dense(
            256,
            name="my_layer2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}

if __name__ == "__main__":
    #args = parser.parse_args()
    ModelCatalog.register_custom_model(
        "keras_model", TestKerasModel)
    ray.init(local_mode=False)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    config = {
        "env": WalkingForwardNorm,  # or "corridor" if registered above
        # "env_config": {
        #     "corridor_length": 5,
        # },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "model": {
            "custom_model": "keras_model",
        #    "vf_share_layers": True,
        },
        # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 11,  # parallelism
        "num_envs_per_worker": 1,
        #"framework": "torch",
        #"num_gpus_per_worker": 0,
        "train_batch_size": 4096*11*4,
        "sgd_minibatch_size": 4096*11*4,
        "rollout_fragment_length": 4096,
        "num_sgd_iter": 10,
    }

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": 2e6,  #args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    results = tune.run("PPO", config=config, stop=stop)

    if False: #args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
