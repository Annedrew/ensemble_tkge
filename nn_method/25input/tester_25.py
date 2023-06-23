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

from constant_25 import *
from trainer_25 import DDataset, NaiveTrainer
from models_nn_25 import NaiveNN, AttentionNN


class NaiveTester:
    def __init__(self):
        pass


    def tester(self, model, trained_model, test_dataset):
        model.load_state_dict(torch.load(trained_model))
        model.eval()
        dataset = DDataset(test_dataset)
        loss_function = nn.MSELoss()
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        with torch.no_grad():
            for inputs, labels in test_loader:
                predicted_output = model(inputs)
                prediction = predicted_output.numpy()
                prediction = pd.DataFrame(prediction)

                mse = loss_function(predicted_output, labels)
                rmse = torch.sqrt(mse)
                prediction.to_csv('nn_method/25input/result/prediction.csv', mode='a', header=False, index=False)
                print(f"Loss: {rmse.item()}")


if __name__ == "__main__":
    model = AttentionNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    dataset_path = "dataset/NN/25t_5w/dataset/test_dataset.csv"

    tester = NaiveTester()
    trained_model = "nn_method/25input/result/best_network.pt"
    tester.tester(model, trained_model, dataset_path)
