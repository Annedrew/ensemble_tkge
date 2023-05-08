import json
import numpy as np
with open("results/icews14/ranked_quads.json", "r") as f:
# with open("temp.json", "r") as f:
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    data = json.load(f)
    rank_values = [list(item['RANK'].values())[:5] for item in data]
    predictions = np.array(rank_values, dtype=int)
    predictions = predictions.transpose()
    predictions = {model_name[i]: predictions[i].tolist() for i in range(len(predictions))}
    for i in range(len(model_name)):
        instance = len(predictions[model_name[i]])
        ranks = sum(predictions[model_name[i]])
        print(f"Number of instance: {instance}")
        print(f"Sumup Ranks ({model_name[i]}): {ranks}")