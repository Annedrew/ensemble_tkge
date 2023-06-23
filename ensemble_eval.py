from loader import Loader
from simulated_facts import SimulatedRank, SimulatedScore, Entity_Id
import json
import ijson
import os
from TERO.rank_calculator import RankCalculator as TERO_Rank
import numpy as np
from eval import Eval
import time

class EnsembleRanking:
    # Calculate ensemble Scores
    def get_ens_score(self, model_name, best_weights, simu_ranks_path):

        ens_scores_head = []
        ens_scores_relation = []
        ens_scores_tail = []
        ens_scores_time = []
        with open(simu_ranks_path, "r") as f:
            objects = ijson.items(f, "item")
            for obj in objects:
                ens_score = 0
                if obj["HEAD"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        # the datatype here is str, not list, using json.loads load the json file into python object
                        # The shape of different query is different, use the short length: [:7129]
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:7129]
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_head.append(ens_score)
                elif obj["RELATION"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:231]
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_relation.append(ens_score)
                elif obj["TAIL"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:7129]
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_tail.append(ens_score)
                elif obj["TIME"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        rank_entity = np.array(json.loads(obj["RANK"][name]))[:7129]
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_time.append(ens_score)

        return ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time


    # Rank all simulated fact by ensemble score
    def get_ens_rank(self, ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time, sim_ranks_path):
        with open(sim_ranks_path, "r") as f:
            data = json.load(f)
            # Enemble Ranking
            ranks_head = []
            ranks_relation = []
            ranks_tail = []
            ranks_time = []
            num_head = 0
            num_relation = 0
            num_tail = 0
            num_time = 0
            for query in range(len(data)):
                if data[query]["HEAD"] == "0":
                    ens_scores_head = np.array(ens_scores_head)
                    # print(ens_scores_head[num_head])
                    # print(f"head time: {num_head}")
                    rank = (ens_scores_head[num_head] < ens_scores_head[num_head][0]).sum() + 1
                    ranks_head.append(rank)
                    # print(f"head ranks: {ranks_head}")
                    num_head += 1
                elif data[query]["RELATION"] == "0":
                    ens_scores_relation = np.array(ens_scores_relation)
                    # print(ens_scores_relation[num_relation])
                    # print(f"relation time: {num_relation}")
                    rank = (ens_scores_relation[num_relation] < ens_scores_relation[num_relation][0]).sum() + 1
                    ranks_relation.append(rank)
                    # print(f"relation ranks: {ranks_relation}")
                    num_relation += 1
                elif data[query]["TAIL"] == "0":
                    ens_scores_tail = np.array(ens_scores_tail)
                    # print(ens_scores_tail[num_tail])
                    # print(f"tail time: {num_tail}")
                    rank = (ens_scores_tail[num_tail] < ens_scores_tail[num_tail][0]).sum() + 1
                    ranks_tail.append(rank)
                    # print(f"tail ranks: {ranks_tail}")
                    num_tail += 1
                elif data[query]["TIME"] == "0":
                    ens_scores_time = np.array(ens_scores_time)
                    # print(ens_scores_time[num_time])
                    # print(f"time time: {num_time}")
                    rank = (ens_scores_time[num_time] < ens_scores_time[num_time][0]).sum() + 1
                    ranks_time.append(rank)
                    # print(f"time ranks: {ranks_time}")
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
        ens_ranks_path = "new_results/ens_rank.json"
        with open(ens_ranks_path, "w") as f:
            json.dump(data, f, indent=4)

        return ens_ranks_path


    def ens_eval(self, dataset_path):
        with open(dataset_path, "r") as f:
            data = json.load(f)
            ens_ranks = []
            for query in range(len(data)):
                ens_ranks.append(json.loads(data[query]["RANK"]))
            eval = Eval()
            ens_eval_path = eval.eval_ens(ens_ranks)
            print(f"The evaluation result saved in {ens_eval_path}")
        return ens_eval_path



if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    # RANK
    # best_weights = [0.5, 0.1, 0.2, 0.1, 0.1]
    # MRR
    # best_weights = [0.1, 0.1, 0.1, 0.3, 0.4]
    # RR
    # best_weights = [0.1, 0.1, 0.1, 0.6, 0.1]
    # Normalized the ensemble score directly
    # best_weights = [0.14, 0.21, 0.19, 0.24, 0.22]
    # ??
    # best_weights = [0.12, 0.19, 0.23, 0.23, 0.23]

    start_time = time.time()
    ens_eval = EnsembleRanking()
    sim_ranks_path = "dataset/ranks/query_ens_test_sim_ranks.json"
    # sim_ranks_path = "dataset/ranks/query_ens_test_sim_ranks.json"
    ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time = ens_eval.get_ens_score(model_name, best_weights, sim_ranks_path)
    print("Ensemble scores has been calculated.")
    ranks_head, ranks_relation, ranks_tail, ranks_time = ens_eval.get_ens_rank(ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time, sim_ranks_path)
    print("Ensemble ranks has been calculated.")
    ens_ranks_path = ens_eval.save_rank(ranks_head, ranks_relation, ranks_tail, ranks_time, sim_ranks_path)
    print("Ensemble ranks has been saved.")
    ens_eval.ens_eval(ens_ranks_path)
    print("Ensemble has been evaluated.")
