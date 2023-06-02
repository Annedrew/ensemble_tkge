from metric_calculator import MetricCalculator
from grid_search import Grid_search
import json
import numpy as np

class Eval():
    # Evaluation for ensemble results
    def eval_ens(self, rank_score: list):
        metric = MetricCalculator()
        sin_rank = {"ENS": rank_score}
    
        eval = metric.calculate_metric(sin_rank)
        with open("new_results/over.json", "w") as f:
            json.dump(eval, f, indent=4)
            
        return eval
    
    
    # Evaluation for individual model result
    def eval_indi(self, rank_score: np.ndarray, model_name):
        metric = MetricCalculator()
        sin_rank = {}
        for i, name in zip(range(len(model_name)), model_name):
            sin_rank[name] = rank_score[i].tolist()
    
        eval = metric.calculate_metric(sin_rank)
        with open("new_results/individual_eval.json", "w") as f:
            json.dump(eval, f, indent=4)
            
        return eval
