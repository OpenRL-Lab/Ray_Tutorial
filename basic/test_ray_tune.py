from ray import tune


def objective(config):
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = {
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

tuner = tune.Tuner(objective, param_space=search_space)

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)