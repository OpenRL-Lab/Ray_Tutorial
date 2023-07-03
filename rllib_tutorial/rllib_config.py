from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import time


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .framework("torch")
    .resources(num_gpus=1)
    .environment(env="CartPole-v1")
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))