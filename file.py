from loader import Loader
from simulated_facts import SimulatedRank, SimulatedScore, Entity_Id
import json
import ijson
import os
from tkge_models.TERO.rank_calculator import RankCalculator as TERO_Rank
import numpy as np
import pandas as pd
from eval import Eval
import time
import csv

class EnsembleRanking():
    def __init__(self):
        pass

    def probability(self, file_path):
        with open(file_path, "r") as f:
            data = pd.read_csv(f)
            for index, row in data.iterrows():
                # Process each row as needed
                print(row)
        
if __name__ == "__main__":
    pro = EnsembleRanking()
    pro.probability("nn_method/5input/result/prediction.csv")
    