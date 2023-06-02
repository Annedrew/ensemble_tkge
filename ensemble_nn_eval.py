from loader import Loader
from simulated_facts import SimulatedRank, SimulatedScore, Entity_Id
import json
import ijson
import os
from TERO.rank_calculator import RankCalculator as TERO_Rank
import numpy as np
from eval import Eval
import time
import csv

class EnsembleRanking():
    def __init__(self):
        pass


    # Calculate ensemble Scores
    def get_ens_score(self, model_name, best_weights, simu_ranks_path):
        with open(simu_ranks_path, "r") as f1, open(best_weights, "r") as f2:
            reader1 = ijson.items(f1, "item")
            reader2 = csv.reader(f2)

            ens_scores_head = []
            ens_scores_relation = []
            ens_scores_tail = []
            ens_scores_time = []
            for obj, best_weights in zip(reader1, reader2):
                ens_score = 0
                if obj["HEAD"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:7129]
                        weight = float(best_weights[i])
                        ens_score += rank_entity * weight / len(model_name)
                    ens_scores_head.append(ens_score)
                elif obj["RELATION"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:231]
                        weight = float(best_weights[i])
                        ens_score += rank_entity * weight / len(model_name)
                    ens_scores_relation.append(ens_score)
                elif obj["TAIL"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:231]
                        weight = float(best_weights[i])
                        ens_score += rank_entity * weight / len(model_name)
                    ens_scores_tail.append(ens_score)
                elif obj["TIME"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:231]
                        weight = float(best_weights[i])
                        ens_score += rank_entity * weight / len(model_name)
                    ens_scores_time.append(ens_score)

        return ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time


    # Rank all simulated fact by ensemble score
    def get_ens_rank(self, ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time, sim_ranks_path):
        with open(sim_ranks_path, "r") as f:
            objects = list(ijson.items(f, "item"))
            # Enemble Ranking
            ranks_head = []
            ranks_relation = []
            ranks_tail = []
            ranks_time = []
            num_head = 0
            num_relation = 0
            num_tail = 0
            num_time = 0

            for obj in objects:
                if obj["HEAD"] == "0":
                    ens_scores_head = np.array(ens_scores_head)
                    rank = (ens_scores_head[num_head] < ens_scores_head[num_head][0]).sum() + 1
                    ranks_head.append(rank)
                    num_head += 1
                elif obj["RELATION"] == "0":
                    ens_scores_relation = np.array(ens_scores_relation)
                    rank = (ens_scores_relation[num_relation] < ens_scores_relation[num_relation][0]).sum() + 1
                    ranks_relation.append(rank)
                    num_relation += 1
                elif obj["TAIL"] == "0":
                    ens_scores_tail = np.array(ens_scores_tail)
                    rank = (ens_scores_tail[num_tail] < ens_scores_tail[num_tail][0]).sum() + 1
                    ranks_tail.append(rank)
                    num_tail += 1
                elif obj["TIME"] == "0":
                    ens_scores_time = np.array(ens_scores_time)
                    rank = (ens_scores_time[num_time] < ens_scores_time[num_time][0]).sum() + 1
                    ranks_time.append(rank)
                    num_time += 1

        return ranks_head, ranks_relation, ranks_tail, ranks_time


    def save_rank(self, ranks_head, ranks_relation, ranks_tail, ranks_time, sim_ranks_path):
        # Write the result into file
        with open(sim_ranks_path, "r") as f:
            data = json.load(f)
            h = 0
            r = 0
            t = 0
            ti = 0
            for query in range(len(data)):
                if data[query]["HEAD"] == "0":
                    data[query]["RANK"] = str(ranks_head[h])
                    h += 1
                elif data[query]["RELATION"] == "0":
                    data[query]["RANK"] = str(ranks_relation[r])
                    r += 1
                elif data[query]["TAIL"] == "0":
                    data[query]["RANK"] = str(ranks_tail[t])
                    t += 1
                elif data[query]["TIME"] == "0":
                    data[query]["RANK"] = str(ranks_time[ti])
                    ti += 1
        ens_ranks_path = "new_results/ensemble_n.json"
        with open(ens_ranks_path, "w") as f:
            json.dump(data, f, indent=4)

        return ens_ranks_path


    def ens_eval(self, best_weights, dataset_path):
        with open(dataset_path, "r") as f:
            data = json.load(f)
            ens_ranks = []
            for query in range(len(data)):
                ens_ranks.append(json.loads(data[query]["RANK"]))
            eval = Eval()
            ens_eval_path = eval.eval_ens(best_weights, ens_ranks)
            print(f"The evaluation result saved in {ens_eval_path}")
        return ens_eval_path



if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    start_time = time.time()
    ens_eval = EnsembleRanking()
    best_weights = "new_results/prediction.csv"
    sim_ranks_path = "dataset/ranks/query_ens_test_sim_ranks.json"
    ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time = ens_eval.get_ens_score(model_name, best_weights, sim_ranks_path)
    print("Ensemble scores has been calculated.")
    ranks_head, ranks_relation, ranks_tail, ranks_time = ens_eval.get_ens_rank(ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time, sim_ranks_path)
    print("Ensemble ranks has been calculated.")
    ens_ranks_path = ens_eval.save_rank(ranks_head, ranks_relation, ranks_tail, ranks_time, sim_ranks_path)
    print("Ensemble ranks has been saved.")
    ens_eval.ens_eval(best_weights, ens_ranks_path)
    print("Ensemble has been evaluated.")
