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


    def load_predictions(self) -> dict:
        with open("results/icews14/ranked_quads.json", 'r') as f:
            data = json.load(f)
            rank_values = [list(item['RANK'].values())[:5] for item in data]
            predictions = np.array(rank_values, dtype=int)
                    
        return predictions


    def dataset_split(self, predictions: dict, args) -> np.ndarray:
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        train, test = train_test_split(predictions, test_size=0.3, random_state=42)
        if args == "rank":
            train = train.transpose()
            test = test.transpose()
        if args == "mrr":
            rank_json = {model_name[i]: train.transpose().tolist()[i] for i in range(len(model_name))}
            metric = MetricCalculator()
            mrr = metric.calculate_metric(rank_json)
            train = []
            for name in model_name:
                train.append(mrr[name]["MRR"])

            rank_json = {model_name[i]: test.transpose().tolist()[i] for i in range(len(model_name))}
            metric = MetricCalculator()
            mrr = metric.calculate_metric(rank_json)
            test = []
            for name in model_name:
                test.append(mrr[name]["MRR"])

        return train, test


    def calculate_ensemble_score(self, rank_score: np.ndarray, model_weights: list):
        ensemble_score = 0
        for j in range(rank_score.shape[1]):
            for i in range(rank_score.shape[0]):
                ensemble_score += model_weights[j] * rank_score[i][j]
                # ensemble_score.append(model_weights[j] * rank_score[i][j])
        ensemble_score = round(ensemble_score, 2)
        print(f"updated ensemble score: {ensemble_score}")

        return ensemble_score
    

    def grid_search_weight(self, rank_score: np.ndarray, model_weights: list, weight_ranges: dict, args):
        # all_score = []
        set_num = 0
        model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
        best_weights = model_weights
        init_score = self.calculate_ensemble_score(rank_score, model_weights)
        # print("init_score: ", init_score)
        grid = itertools.product(*list(weight_ranges.values()))
        for weights in grid:
            if sum(weights) != 1:
                pass
            else:
                set_num += 1
                if args == "rank":
                    ensemble_score = self.calculate_ensemble_score(rank_score, list(weights))
                    if ensemble_score < init_score:
                        init_score = ensemble_score
                        best_weights = list(weights)
                    # all_score.append(init_score)
                    print(f"updated weight: {weights}")
                elif args == "mrr":
                    ensemble_score = self.calculate_ensemble_score(rank_score, list(weights))
                    if ensemble_score > init_score:
                        init_score = ensemble_score
                        best_weights = list(weights)
                    # all_score.append(init_score)
                    print(f"updated weight: {weights}")
        metric = 0
        mrr = 0
        if args == "rank":
            for i in range(len(best_weights)):
                for j in range(rank_score.shape[0]):
                    mrr += (1 / rank_score[j][i]) 
                metric += (mrr / rank_score.shape[0]) * best_weights[i]
            print(f"metric: {metric}")
        elif args == "mrr":
            for i in range(len(best_weights)):
                for j in range(rank_score.shape[0]):
                    metric += rank_score[j][i] * best_weights[i]
            print(f"metric: {metric}")

        # print(f"rank score: {rank_score}")
        # print(f"The number of qualified sets: {set_num}")
        print(f"class best weight: {best_weights}")
        # all_score.sort()
        # print(f"The list of ensemble score: {all_score}")

        return best_weights, metric
    

