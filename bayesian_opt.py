from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from grid_search import *
from constants import *
from arguments import *
from metric_calculator import MetricCalculator


class Bayesian_opt:
    def __init__(self):
        pass


    def objective_function(self, w1, w2, w3, w4, w5):
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        weights = [w1, w2, w3, w4, w5]

        # get the training dataset
        ens_train = []
        with open("ens_train.json", "r") as f:
            data = json.load(f)
            for query in range(len(data)):
                # Remember to filter out TFLEX model
                rank = [int(item) for item in data[query]["RANK"].values()][:5]
                ens_train.append(rank)
            ens_train = np.array(ens_train)
            # shape: (5, 25096)
            ens_train = ens_train.transpose()

        # get the mrr score
        metric = MetricCalculator()
        rank_json = {model_name[i]: ens_train.tolist()[i] for i in range(len(model_name))}
        mrr = metric.calculate_metric(rank_json)
        mrr_score = []
        hit1_score = []
        for name in model_name:
            mrr_score.append([mrr[name]["MRR"]])
            hit1_score.append([mrr[name]["Hits@1"]])
        mrr_score = np.array(mrr_score)
        hit1_score = np.array(hit1_score)
        print(mrr_score.shape)
        print(hit1_score.shape)

        # get the ensemble_score_mrr
        ensemble_score_mrr = 0
        for j in range(mrr_score.shape[1]):
            for i in range(mrr_score.shape[0]):
                ensemble_score_mrr += weights[i] * mrr_score[i][j]

        # get the ensemble_score_hit1
        ensemble_score_hit1 = 0
        for j in range(mrr_score.shape[1]):
            for i in range(mrr_score.shape[0]):
                ensemble_score_hit1 += weights[i] * mrr_score[i][j]
        
        return ensemble_score_mrr#, ensemble_score_hit1
    

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
        best_weights = optimizer.max['params']
        
        return best_weights