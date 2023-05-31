# ALL CLASS:
# SumRank - Analysis the sum of ranks as metric
# CorrectRange - Analysis the concentrated range of correct answer
# Duplicates - Analysis the shared entity/relation/time in the concentrated range
# NeighborDiff - Analysis the difference of scores between each two neighbor rank

import json
import ijson
import csv
import numpy as np
import pandas as pd


# Analysis the sum of ranks as metric
class SumRank:
    def __init__(self):
        pass
    

    def sum_rank(self, file_path, model_name):
        with open(file_path, "r") as f:
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


# Analysis the concentrated range of correct answer
class CorrectRange():
    def __init__(self):
        pass


    # Get the rank for each model
    def ranks_model(self, file_path, model_name):
        de_transe = []
        de_simple = []
        de_distmult = []
        tero = []
        atise = []
        with open(file_path, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                for name in model_name:
                    if name == "DE_TransE":
                        de_transe.append(data[i]["RANK"][name])
                    elif name == "DE_SimplE":
                        de_simple.append(data[i]["RANK"][name])
                    elif name == "DE_DistMult":
                        de_distmult.append(data[i]["RANK"][name])
                    elif name == "TERO":
                        tero.append(data[i]["RANK"][name])
                    elif name == "ATISE":
                        atise.append(data[i]["RANK"][name])
            
        # Convert into numpy array
        de_transe = np.array(de_transe)
        de_simple = np.array(de_simple)
        de_distmult = np.array(de_distmult)
        tero = np.array(tero)
        atise = np.array(atise)

        return de_transe, de_simple, de_distmult, tero, atise

    
    def rank_unique(self, de_transe, de_simple, de_distmult, tero, atise):
        # Take the unique
        de_transe_rank, de_transe_counts = np.unique(de_transe, return_counts=True)
        de_simple_rank, de_simple_counts = np.unique(de_simple, return_counts=True)
        de_distmult_rank, de_distmult_counts = np.unique(de_distmult, return_counts=True)
        tero_rank, tero_counts = np.unique(tero, return_counts=True)
        atise_rank, atise_counts = np.unique(atise, return_counts=True)

        # Save csv
        with open("range_concentrate.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["DE_TransE_Rank", "DE_TransE_Counts", "DE_SimplE_Rank", "DE_SimplE_Counts", "DE_DistMult_Rank", "DE_DistMult_Counts", "TERO_Rank", "TERO_Counts", "ATISE_Rank", "ATISE_Counts"])
            for rank1, count1, rank2, count2, rank3, count3, rank4, count4, rank5, count5 in zip(de_transe_rank, de_transe_counts, de_simple_rank, de_simple_counts, de_distmult_rank, de_distmult_counts, tero_rank, tero_counts, atise_rank, atise_counts):
                writer.writerow([rank1, count1, rank2, count2, rank3, count3, rank4, count4, rank5, count5])


    def order_csv(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        sorted_rows = list(sorted(rows, key=lambda row: int(row[1]), reverse=True))
        with open(f"ordered_{file_path}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(sorted_rows)


# Analysis the shared entity/relation/time in the concentrated range
class Duplicates:
    def __init__(self):
        pass


    # Duplicate in total
    # can be used to get target too
    def detect_dupli(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            duplicates = []
            for row in rows:
                row = [int(i) for i in row]
                row = np.array(row)
                id, counts = np.unique(row, return_counts=True)
                duplicate = id[counts > 1]
                if len(duplicate) == 0:
                    # -1 means no duplicates for this query
                    duplicate = [-1]
                duplicates.append(duplicate)

        return duplicates
    

    def save_csv(self, file_path, duplicates):
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            for row in duplicates:
                writer.writerow(row)


# Analysis the difference of scores between each two neighbor rank
class NeighborDiff:
    def __init__(self):
        pass


    def difference(self, file_path, model_name):
        sorted_data = {}
        for name in model_name:
            sorted_data[name] = []

        with open(file_path, "r") as f:
            objects = ijson.items(f, "item")
            for obj in objects:
                # TODO: r, t and T can also be implemented
                if obj["HEAD"] == "0":
                    if model_name[0] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[0]]))
                        differences_0 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[1] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[1]]))
                        differences_1 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[2] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[2]]))
                        differences_2 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[3] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[3]]))
                        differences_3 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[4] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[4]]))
                        differences_4 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    filename = 'new_results/h_diff.csv'
                    with open(filename, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Difference_0', 'Difference_1', 'Difference_2', 'Difference_3', 'Difference_4'])
                        writer.writerows([list(diff) for diff in zip(differences_0, differences_1, differences_2, differences_3, differences_4)])


    # 统计排名在前100的问题有多少个，是整个test集合的
    def hit_100(self):
        extracted_values = []
        with open("results/icews14/ranked_quads.json", "r") as f:
            data = json.load(f)

            transe = []
            simple = []
            distmult = []
            tero = []
            atise = []
            for query in range(len(data)):
                transe.append(json.loads(data[query]['RANK']['DE_TransE']))
                simple.append(json.loads(data[query]['RANK']['DE_SimplE']))
                distmult.append(json.loads(data[query]['RANK']['DE_DistMult']))
                tero.append(json.loads(data[query]['RANK']['TERO']))
                atise.append(json.loads(data[query]['RANK']['ATISE']))
            # TransE: 35852
            # SimplE: 35852
            # DistMult: 35852
            # TeRo: 35852
            # ATiSe: 35852
            print(f"TransE: {len(transe)}")
            print(f"SimplE: {len(simple)}")
            print(f"DistMult: {len(distmult)}")
            print(f"TeRo: {len(tero)}")
            print(f"ATiSe: {len(atise)}\n")
                
            transe1 = [rank for rank in transe if rank < 100]
            simple1 = [rank for rank in simple if rank < 100]
            distmult1 = [rank for rank in distmult if rank < 100]
            tero1 = [rank for rank in tero if rank < 100]
            atise1 = [rank for rank in atise if rank < 100]

            # TransE(>100): 28828
            # SimplE(>100): 27559
            # DistMult(>100): 27117
            # TeRo(>100): 28865
            # ATiSe(>100): 28802
            print(f"TransE(>100): {len(transe1)}")
            print(f"SimplE(>100): {len(simple1)}")
            print(f"DistMult(>100): {len(distmult1)}")
            print(f"TeRo(>100): {len(tero1)}")
            print(f"ATiSe(>100): {len(atise1)}\n")

            # TransE(>100): 0.8040834542006025
            # SimplE(>100): 0.768687939306036
            # DistMult(>100): 0.7563594778533973
            # TeRo(>100): 0.8051154747294432
            # ATiSe(>100): 0.8033582505857414
            print(f"TransE(>100): {len(transe1)/len(transe)}")
            print(f"SimplE(>100): {len(simple1)/len(simple)}")
            print(f"DistMult(>100): {len(distmult1)/len(distmult)}")
            print(f"TeRo(>100): {len(tero1)/len(tero)}")
            print(f"ATiSe(>100): {len(atise1)/len(atise)}")




        

if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    dataset = NeighborDiff()
    dataset.hit_100()
    # Decide to try sum of ranks as metric
    # sumrank = SumRank()
    # sumrank.sum_rank("results/icews14/ranked_quads.json", model_name)

    # Decide the input range for each model
    # range = CorrectRange()
    # de_transe, de_simple, de_distmult, tero, atise = range.ranks_model("results/icews14/ranked_quads.json", model_name)
    # range.rank_unique(de_transe, de_simple, de_distmult, tero, atise)
    # range.order_csv("range_concentrate.csv")

    # Decide to use score or one-hot-encoding as input and output
    # duplicate = Duplicates()
    # duplicates = duplicate.detect_dupli("new_results/temp_top_5_id.csv")
    # duplicate.save_csv("new_results/duplicates_temp_top_5_id.csv", duplicates)
    # duplicates = duplicate.detect_dupli("new_results/ens_train_top_5_id.csv")
    # duplicate.save_csv("new_results/duplicates_ens_train_top_5_id.csv", duplicates)

    # Decide to use score or one-hot-encoding as output
    # diff = NeighborDiff()
    # diff.difference("new_results/temp_sim_scores.json", model_name)