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
from models_nn import NaiveNN, AttentionNN


class NaiveTester:
    def __init__(self):
        pass


    def tester(self, model, test_dataset):
        model.load_state_dict(torch.load('nn_method/my_baby.pt'))
        model.eval()
        dataset = DDataset(test_dataset)
        loss_function = nn.MSELoss()
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                predicted_output = model(inputs)
                predicted_output = predicted_output.numpy()
                prediction = pd.DataFrame(predicted_output)

                mse = loss_function(outputs, batch_labels)
                rmse = torch.sqrt(mse)
                prediction.to_csv('new_results/prediction.csv', mode='a', header=False, index=False)
                print(f"Loss: {rmse.item()}")


if __name__ == "__main__":
    model = AttentionNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    dataset_path = "dataset/NN/5p_5w/test_dataset.csv"

    tester = NaiveTester()
    tester.tester(model, dataset_path)
