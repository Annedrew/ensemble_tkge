from bayes_opt import BayesianOptimization
from grid_search import *


class Bayesian_opt:
    def __init__(self):
        pass


    def objective_function(self, w1, w2, w3, w4, w5):
        gs = Grid_search()
        predictions = gs.load_predictions("result/icews14/ranked_quads.json")
        ens_train, ens_test = gs.dataset_split(predictions)
        ensemble_score = 0
        for i in range(ens_train.shape[0]):
            ensemble_score = w1 * ens_train[i][0] + w2 * ens_train[i][1] + w3 * ens_train[i][2] + w4 * ens_train[i][3] + w5 * ens_train[i][4]
            ensemble_score = -round(ensemble_score, 2)
            # print(ensemble_score)
                
        return ensemble_score


    def bayesian_opt_weight(self):#, bounds=BOUNDS):
        BOUNDS = {'w1': (0.5, 1.0), 'w2': (0.1, 0.5), 'w3': (0.2, 0.5), 'w4': (0.1, 0.5), 'w5': (0.1, 0.5)}

        optimizer = BayesianOptimization(
            f=self.objective_function, 
            pbounds=BOUNDS,
            # for reproducibility
            random_state=1,
            allow_duplicate_points=True
        )
        optimizer.maximize(
            init_points=5,
            n_iter=10
        )
        print(f"Ensemble Score: {optimizer.max['target']:.4f}")
        print(f"Best weights: {optimizer.max['params']}")
        