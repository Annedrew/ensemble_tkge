from grid_search import Grid_search
from bayesian_opt import Bayesian_opt
import time
from arguments import *
from constants import *
from save_result import *

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
        predictions = my_grid_search.load_predictions("/Users/dutao/Documents/github/ensemble_tkge/results/icews14/ranked_quads.json")
        # predictions = my_grid_search.load_predictions("results/icews14/ranked_quads.json")
        ens_train, ens_test = my_grid_search.dataset_split(predictions)

        start_time = time.time()
        best_weights = my_grid_search.grid_search_weight(ens_train, list(init_weight.values()), weight_ranges)
        print("best weight: ", best_weights)

        ensemble_score = my_grid_search.calculate_ensemble_score(ens_train, best_weights)
        print("ensemble_score: ", ensemble_score)
        time_grid = time.time()
        run_time = round((time_grid - start_time), 2)
        
    # Bayesian Optimization
    elif args.method == "bayes":
        start_time = time.time()
        my_bayesian_opt = Bayesian_opt()
        best_weights = my_bayesian_opt.bayesian_opt_weight()
        ensemble_score = my_bayesian_opt.objective_function(best_weights[0], best_weights[1], best_weights[2], best_weights[3], best_weights[4])
        time_bayes = time.time()
        run_time = round((time_bayes - start_time), 2)
        
    save_file(best_weights, run_time, ensemble_score, args)