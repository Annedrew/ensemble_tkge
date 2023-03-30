import itertools
import json
import numpy as np
from constants import *

class grid_search:
    def __init__(self):
        pass


    def load_predictions(self, file_path) -> dict:
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        with open(file_path, 'r') as f:
            data = json.load(f)
            rank_values = [list(item['RANK'].values())[:5] for item in data]
            predictions = np.array(rank_values, dtype=int)
            predictions = predictions.transpose()
            predictions = {model_name[i]: predictions[i].tolist() for i in range(len(predictions))}
            
        return predictions


    def calculate_ensemble_score(self, y_pred, models):
        predictions = self.load_predictions('result/icews14/ranked_quads.json')
        ensemble_score = 0
        for model_name in models:
            weight = models[model_name]
            for i in range(y_pred.ndim):
                ensemble_score += weight * predictions[model_name][i]
        ensemble_score = round(ensemble_score, 2)

        return ensemble_score
    

    def grid_search_weight(self, y_pred, models, weight_ranges):
        best_weights = None
        init_score = self.calculate_ensemble_score(y_pred, models)
        print("init_score: ", init_score)
        grid = itertools.product(*[weight_ranges[model] for model in models])
        for weights in grid:
            if sum(weights) != 1:
                pass
            else:
                for i, model_name in enumerate(models):
                    models[model_name] = weights[i]
                    ensemble_score = self.calculate_ensemble_score(y_pred, models)
                    if ensemble_score < init_score:
                        init_score = ensemble_score
                        best_weights = weights
                    else:
                        best_weights = list(models.values())
        
        return best_weights

