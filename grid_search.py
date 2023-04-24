import itertools
import json
import numpy as np
from sklearn.model_selection import train_test_split
from constants import *
from metric_calculator import MetricCalculator
from arguments import *

class Grid_search:
    def __init__(self):
        pass


    def load_predictions(self) -> np.ndarray:
        with open("results/icews14/ranked_quads.json", 'r') as f:
            data = json.load(f)
            rank_values = [list(item['RANK'].values())[:5] for item in data]
            predictions = np.array(rank_values, dtype=int)
                    
        return predictions


    def dataset_split(self, predictions: np.ndarray) -> np.ndarray:
        train, test = train_test_split(predictions, test_size=0.3, random_state=42)
        train = train.transpose()
        test = test.transpose()

        return train, test


    def calculate_ensemble_score(self, rank_score: np.ndarray, model_weights: list):
        ensemble_score = 0
        for j in range(rank_score.shape[1]):
            for i in range(rank_score.shape[0]):
                ensemble_score += model_weights[i] * rank_score[i][j]
        ensemble_score = round(ensemble_score, 2)
        print(f"updated ensemble score: {ensemble_score}")

        return ensemble_score


    def grid_search_weight(self, rank_score: np.ndarray, model_weights: list, weight_ranges: dict, args):
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        best_weights = model_weights
        if args == "rank":
            init_score = self.calculate_ensemble_score(rank_score, model_weights)
            grid = itertools.product(*list(weight_ranges.values()))
            for weights in grid:
                if sum(weights) != 1:
                    pass
                else:
                    ensemble_score = self.calculate_ensemble_score(rank_score, list(weights))
                    if ensemble_score < init_score:
                        init_score = ensemble_score
                        best_weights = list(weights)
                    print(f"updated weight: {weights}")
        elif args == "mrr":
            rank_json = {model_name[i]: rank_score.transpose().tolist()[i] for i in range(len(model_name))}
            metric = MetricCalculator()
            mrr = metric.calculate_metric(rank_json)
            mrr_score = []
            for name in model_name:
                mrr_score.append([mrr[name]["MRR"]])
            mrr_score = np.array(mrr_score)

            init_score = self.calculate_ensemble_score(mrr_score, model_weights)
            grid = itertools.product(*list(weight_ranges.values()))
            for weights in grid:
                if sum(weights) != 1:
                    pass
                else:
                    ensemble_score = self.calculate_ensemble_score(mrr_score, list(weights))
                    if ensemble_score > init_score:
                        init_score = ensemble_score
                        best_weights = list(weights)
                    print(f"updated weight: {weights}")

        return best_weights
    

