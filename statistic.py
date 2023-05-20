# To analysis the concentrated range of correct answer

import json
import csv
import numpy as np

class Statistic():
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


# Count the number of ranks
if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    statistic = Statistic()
    de_transe, de_simple, de_distmult, tero, atise = statistic.ranks_model("results/icews14/ranked_quads.json", model_name)
    statistic.rank_unique(de_transe, de_simple, de_distmult, tero, atise)
    statistic.order_csv("range_concentrate.csv")