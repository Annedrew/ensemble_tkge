from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from grid_search import *
from constants import *


class Bayesian_opt:
    def __init__(self):
        pass


    def objective_function(self, w1, w2, w3, w4, w5):
        weights = [w1, w2, w3, w4, w5]
        gs = Grid_search()
        predictions = gs.load_predictions("results/icews14/ranked_quads.json")
        # predictions = gs.load_predictions("temp.json")
        ens_train, ens_test = gs.dataset_split(predictions)
        ensemble_score = 0
        if sum(weights) != 1:
            pass
        else:
            for i in range(ens_train.shape[0]):
                for j, k in zip(range(ens_train.shape[1]), range(len(weights))):
                    ensemble_score += weights[k] * ens_train[i][j]
        ensemble_score = - ensemble_score
        print(ensemble_score)
        
        return ensemble_score
    

    def bayesian_opt_weight(self, bounds=BOUNDS):
        optimizer = BayesianOptimization(
            f=self.objective_function, 
            pbounds=bounds,
            # for reproducibility
            random_state=1,
            allow_duplicate_points=True
        )

        optimizer.maximize(
            init_points=5,
            n_iter=10
        )

        # Continue the optimization process from a previously saved state
        # optimizer = load_logs(optimizer, logs=["./logs_bayes.json"])

        print(f"Ensemble Score: {optimizer.max['target']:.4f}")
        # print(f"Best weights: {optimizer.max['params']}")
        best_weights = optimizer.max['params']
        return best_weights