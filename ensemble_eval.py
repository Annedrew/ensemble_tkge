from loader import Loader
from ranker import Ranker
import json
import os
from TERO.rank_calculator import RankCalculator as TERO_Rank
import numpy as np
from eval import Eval
import time

class EnsembleRanking():
    # Get ensemble rankings for all simulated fact
    def load_model(self, model_name, file_path):
        with open(file_path, "r") as f:
            ranked_quads = json.load(f)
            for name in model_name:
                model_path = os.path.join("models", name, "icews14", "Model.model")
                loader = Loader(model_path, name)
                model = loader.load()
                ranker = Ranker(ranked_quads, model, name)
                #  Get a list of all the rankings, not only correct ranking
                rank = ranker.rank()
        with open("a.json", "w") as f:
            json.dump(rank, f, indent=4)


    # Calculate ensemble Scores
    def get_ens_score(self, model_name, best_weights):
        with open("a.json", "r") as f:
            data = json.load(f)
            ens_scores_head = []
            ens_scores_relation = []
            ens_scores_tail = []
            ens_scores_time = []
            for query in range(len(data)):
                ens_score = 0
                if data[query]["HEAD"] == "0":
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        # The datatype here is str, not list, using json.loads load the json file into python object
                        # The shape of different query is different, use the short length: [:7129]
                        # if name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                        rank_entity = np.array(json.loads(data[query]["RANK"][name]))[:7129]
                        ens_score += rank_entity * best_weights[i]
                        # elif name in ["TERO", "ATISE"]:
                        #     rank_entity = np.array(json.loads(data[query]["RANK"][name]))
                        #     ens_score += rank_entity * best_weights[i]
                    ens_scores_head.append(ens_score)
                elif data[query]["RELATION"] == "0":
                    # The ensemble score for 1 query
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        # the datatype here is str, not list, using json.loads load the json file into python object
                        # The shape of different query is different, use the short length: [:7129]
                        rank_entity = np.array(json.loads(data[query]["RANK"][name]))[:231]
                        # print(f"shape: {rank_entity.shape}")
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_relation.append(ens_score)
                elif data[query]["TAIL"] == "0":
                    # The ensemble score for 1 query
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        # the datatype here is str, not list, using json.loads load the json file into python object
                        # The shape of different query is different, use the short length: [:7129]
                        rank_entity = np.array(json.loads(data[query]["RANK"][name]))[:7129]
                        # print(f"shape: {rank_entity.shape}")
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_tail.append(ens_score)
                elif data[query]["TIME"] == "0":
                    # The ensemble score for 1 query
                    for i, name in zip(range(len(model_name)), model_name):
                        # The rank of all entities
                        # the datatype here is str, not list, using json.loads load the json file into python object
                        # The shape of different query is different, use the short length: [:7129]
                        rank_entity = np.array(json.loads(data[query]["RANK"][name]))
                    #     print(f"shape: {rank_entity.shape}")
                        ens_score += rank_entity * best_weights[i]
                    ens_scores_time.append(ens_score)

        return ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time

    # Rank all simulated fact by ensemble score
    def get_ens_rank(self, ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time):
        with open("a.json", "r") as f:
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
                    print(f"head ranks: {ranks_head}")
                    num_head += 1
                elif data[query]["RELATION"] == "0":
                    ens_scores_relation = np.array(ens_scores_relation)
                    # print(ens_scores_relation[num_relation])
                    # print(f"relation time: {num_relation}")
                    rank = (ens_scores_relation[num_relation] < ens_scores_relation[num_relation][0]).sum() + 1
                    ranks_relation.append(rank)
                    print(f"relation ranks: {ranks_relation}")
                    num_relation += 1
                elif data[query]["TAIL"] == "0":
                    ens_scores_tail = np.array(ens_scores_tail)
                    # print(ens_scores_tail[num_tail])
                    # print(f"tail time: {num_tail}")
                    rank = (ens_scores_tail[num_tail] < ens_scores_tail[num_tail][0]).sum() + 1
                    ranks_tail.append(rank)
                    print(f"tail ranks: {ranks_tail}")
                    num_tail += 1
                elif data[query]["TIME"] == "0":
                    ens_scores_time = np.array(ens_scores_time)
                    # print(ens_scores_time[num_time])
                    # print(f"time time: {num_time}")
                    rank = (ens_scores_time[num_time] < ens_scores_time[num_time][0]).sum() + 1
                    ranks_time.append(rank)
                    print(f"time ranks: {ranks_time}")
                    num_time += 1

        return ranks_head, ranks_relation, ranks_tail, ranks_time


    def save_rank(self, ranks_head, ranks_relation, ranks_tail, ranks_time):
        # Write the result into file
        with open("a.json", "r") as f:
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
        with open("ensemble_n.json", "w") as f:
            json.dump(data, f, indent=4)


    def ens_eval(self, best_weights):
        with open("a.json", "r") as f:
            data = json.load(f)
            ens_ranks = []
            for query in range(len(data)):
                ens_ranks.append(json.loads(data[query]["RANK"]))
            eval = Eval()
            eval.eval_ens(best_weights, ens_ranks)



if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    best_weights = [0.14, 0.21, 0.19, 0.24, 0.22]
    # best_weights = [0.5, 0.1, 0.2, 0.1, 0.1]
    # best_weights = [0.1, 0.1, 0.1, 0.3, 0.4]
    # best_weights = [0.1, 0.1, 0.1, 0.6, 0.1]

    start_time = time.time()
    ens_eval = EnsembleRanking()
    ens_eval.load_model(model_name, "ens_test.json")
    ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time = ens_eval.get_ens_score(model_name, best_weights)
    ranks_head, ranks_relation, ranks_tail, ranks_time = ens_eval.get_ens_rank(ens_scores_head, ens_scores_relation, ens_scores_tail, ens_scores_time)
    ens_eval.save_rank(ranks_head, ranks_relation, ranks_tail, ranks_time)
    ens_eval.ens_eval(best_weights)
