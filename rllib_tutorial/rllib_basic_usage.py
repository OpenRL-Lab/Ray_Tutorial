import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# train
# algo = (
#     PPOConfig()
#     .rollouts(num_rollout_workers=1)
#     .framework("torch")
#     .resources(num_gpus=0)
#     .environment(env="CartPole-v1")
#     .build()
# )
#
# for i in range(10):
#     result = algo.train()
#     print(pretty_print(result))
#
#     if i % 5 == 0:
#         checkpoint_dir = algo.save()
#         print(f"Checkpoint saved in directory {checkpoint_dir}")


# tune
ray.init()

config = PPOConfig()\
    .training(lr=tune.grid_search([0.01, 0.001, 0.0001]))\
    .framework("torch")\
    .environment(env="CartPole-v1")

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 150},
        checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
    ),
    param_space=config,
)

results = tuner.fit()

# Get the best result based on a particular metric.
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint

# load checkpoint
# from ray.rllib.algorithms.algorithm import Algorithm
# algo = Algorithm.from_checkpoint(checkpoint_path)