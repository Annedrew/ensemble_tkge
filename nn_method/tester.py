import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from torch.utils.data import DataLoader

# Fix the error: modulenot...
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ensemble_tkge.constants import *
from trainer import DDataset, NaiveTrainer
from models_nn import NaiveNN


class NaiveTester:
    def __init__(self):
        pass


    def tester(self, model, test_dataset):
        model.load_state_dict(torch.load('nn_method/my_baby.pt'))
        model.eval()
        dataset = DDataset(test_dataset)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_function = nn.MSELoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                predicted_output = model(inputs)
                predicted_output = predicted_output.numpy()
                prediction = pd.DataFrame(predicted_output)
                prediction.to_csv('new_results/prediction.csv', mode='a', header=False, index=False)

if __name__ == "__main__":
    model = NaiveNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    dataset_path = "ens_test_h5_h5_norm.csv"

    tester = NaiveTester()
    tester.tester(model, dataset_path)
