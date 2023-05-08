from metric_calculator import MetricCalculator
from grid_search import Grid_search
import json
import numpy as np

class Eval():
    # Evaluation for ensemble results
    def eval_ens(self, best_weights, rank_score: list):
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        metric = MetricCalculator()
        # for model in rank_score:
        sin_rank = {"ENS": rank_score}
    
        eval = metric.calculate_metric(sin_rank)
        with open("over.json", "w") as f:
            json.dump(eval, f, indent=4)
            
        return eval
    
    # Old evaluation for ensemble
    def eval_ens_old(self, best_weights, rank_score):
        metric = MetricCalculator()
        ensemble_scores = []
        for i in range(rank_score.shape[1]):
            ensemble_score = 0
            for j in range(rank_score.shape[0]):
                ensemble_score += best_weights[j] * rank_score[j][i]
            ensemble_scores.append(ensemble_score)
        ens_rank = {"ens": ensemble_scores}
        # print(ens_rank)
        eval_ens = metric.calculate_metric(ens_rank)
        # eval_ens = eval_ens.values()
        # eval_ens = ', '.join(str(value) for value in eval_ens)
        with open("over_ens.json", "w") as f:
            json.dump(eval_ens, f, indent=4)

        return eval_ens
    
    # Evaluation for individual model result
    def eval_indi(self, rank_score: np.ndarray):
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        metric = MetricCalculator()
        # for model in rank_score:
        sin_rank = {"DE_TransE": rank_score[0].tolist(), "DE_SimplE": rank_score[1].tolist(), "DE_DistMult": rank_score[2].tolist(), "TERO": rank_score[3].tolist(), "ATISE": rank_score[4].tolist()}
    
        eval = metric.calculate_metric(sin_rank)
        with open("individual_eval.json", "w") as f:
            json.dump(eval, f, indent=4)
            
        return eval
