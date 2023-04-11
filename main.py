from grid_search import Grid_search
from bayesian_opt import Bayesian_opt
import numpy as np
import json
from sklearn.model_selection import train_test_split
import time


if __name__ == "__main__":
    # Grid Search
    init_weight = {
        'DE_TransE': 0.2,
        'DE_SimplE': 0.2,
        'DE_DistMult': 0.2,
        'TERO': 0.2,
        'ATISE': 0.2
    }

    weight_ranges = {
        'DE_TransE':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'DE_SimplE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'DE_DistMult': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'TERO': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'ATISE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }


    my_grid_search = Grid_search()
    predictions = my_grid_search.load_predictions("result/icews14/ranked_quads.json")
    ens_train, ens_test = my_grid_search.dataset_split(predictions)
    print(f"train: {ens_train}")
    print(f"test: {ens_test}")
    start_time = time.time()
    # best_weights = my_grid_search.grid_search_weight(ens_train, list(init_weight.values()), weight_ranges)
    # print("best weight: ", best_weights)

    # ensemble_score = my_grid_search.calculate_ensemble_score(ens_train, best_weights)
    # print("ensemble_score: ", ensemble_score)
    time_grid = time.time()

    # Bayesian Optimization
    my_bayesian_opt = Bayesian_opt()
    best_weights = my_bayesian_opt.bayesian_opt_weight()
    time_bayes = time.time()
    print(f"The running time for grid search: {time_grid - start_time}")
    print(f"The running time for bayesian optimization search: {time_bayes - time_grid}")
