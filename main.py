from grid_search import Grid_search
from bayesian_opt import Bayesian_opt
import time
from arguments import *
from constants import *

if __name__ == "__main__":
    args = init_args()
    # Grid Search
    if args.method == "grid":
        init_weight = {
            'DE_TransE': 0.2,
            'DE_SimplE': 0.2,
            'DE_DistMult': 0.2,
            'TERO': 0.2,
            'ATISE': 0.2
        }
        my_grid_search = Grid_search()
        predictions = my_grid_search.load_predictions("result/icews14/ranked_quads.json")
        ens_train, ens_test = my_grid_search.dataset_split(predictions)

        start_time = time.time()
        best_weights = my_grid_search.grid_search_weight(ens_train, list(init_weight.values()), weight_ranges)
        print("best weight: ", best_weights)

        ensemble_score = my_grid_search.calculate_ensemble_score(ens_train, best_weights)
        print("ensemble_score: ", ensemble_score)
        time_grid = time.time()
        print(f"The running time for grid search: {round((time_grid - start_time), 2)}s")
    # Bayesian Optimization
    elif args.method == "bayes":
        start_time = time.time()
        my_bayesian_opt = Bayesian_opt()
        best_weights = my_bayesian_opt.bayesian_opt_weight()
        time_bayes = time.time()
        print(f"The running time for bayesian optimization search: {round((time_bayes - start_time), 2)}s")
