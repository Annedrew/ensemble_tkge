from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from grid_search import *
from constants import *
from arguments import *


class Bayesian_opt:
    def __init__(self):
        pass


    def objective_function(self, w1, w2, w3, w4, w5):
        args = init_args()
        weights = [w1, w2, w3, w4, w5]
        gs = Grid_search()
        predictions = gs.load_predictions(args.metric)
        # predictions = gs.load_predictions("temp.json")
        ens_train, ens_test = gs.dataset_split(predictions, args.metric)
        ensemble_score = 0
        # if sum(weights) != 1:
        #     ensemble_score = 0
        # else:
        for i in range(ens_train.shape[0]):
            for j in range(len(weights)):
                ensemble_score += weights[j] * ens_train[i][j]
        ensemble_score = ensemble_score
        metric = 0
        for i in range(len(weights)):
            metric += ens_train[0][i] * weights[i]
        
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