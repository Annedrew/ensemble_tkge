from grid_search import grid_search
import numpy as np


if __name__ == "__main__":
    # Grid Search
    models = {
        'DE_TransE': 0.2,
        'DE_SimplE': 0.2,
        'DE_DistMult': 0.2,
        'TERO': 0.2,
        'ATISE': 0.2
    }

    weight_ranges = {
        'DE_TransE':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        'DE_SimplE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        'DE_DistMult': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        'TERO': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        'ATISE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    }

    my_grid_search = grid_search()
    y_pred = np.array(list((my_grid_search.load_predictions('temp.json')).values()))
    best_weights = my_grid_search.grid_search_weight(y_pred, models, weight_ranges)
    print("best weight: ", best_weights)
    ensemble_score = my_grid_search.calculate_ensemble_score(y_pred, {model_name: best_weights[i] for i, model_name in enumerate(models)})
    print("ensemble_score: ", ensemble_score)
