import itertools
import json
import numpy as np
from constants import *
from sklearn.model_selection import train_test_split

class Grid_search:
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
    

    def dataset_split(self, predictions: dict) -> list:
        predictions = np.transpose(np.array(list(predictions.values())))
        train, test = train_test_split(predictions, test_size=0.3, random_state=42)

        return train, test


    def calculate_ensemble_score(self, rank_score: np.ndarray, model_weights: list):
        ensemble_score = 0
        for i in range(rank_score.shape[0]):
            for j in range(rank_score.shape[1]):
                ensemble_score += model_weights[j] * rank_score[i][j]
        ensemble_score = round(ensemble_score, 2)
        print(f"updated ensemble score: {ensemble_score}")

        return ensemble_score
    

    def grid_search_weight(self, rank_score: np.ndarray, model_weights: list, weight_ranges: dict):
        best_weights = None
        init_score = self.calculate_ensemble_score(rank_score, model_weights)
        print("init_score: ", init_score)
        grid = itertools.product(*list(weight_ranges.values()))
        for weights in grid:
            if sum(weights) != 1:
                pass
            else:
                ensemble_score = self.calculate_ensemble_score(rank_score, list(weights))
                if ensemble_score < init_score:
                    init_score = ensemble_score
                    model_weights = list(weights)
                    best_weights = model_weights
                else:
                    best_weights = model_weights
                print(f"updated weight: {best_weights}")
        
        return best_weights
    

