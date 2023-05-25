import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

# Fix the error: modulenot...
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nn_models import NaiveNN
from ensemble_tkge.constants import *

import torch
from torch.utils.data import Dataset
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)


    def __getitem__(self, index):
        # The weight matrix of nn.Linear using float32, so there must be float32
        input_sample = self.data.iloc[index, :INPUT_SIZE].values.astype(np.float32)
        target_label = self.data.iloc[index, - OUTPUT_SIZE:].values.astype(np.float32)

        input_sample = torch.tensor(input_sample)
        target_label = torch.tensor(target_label)

        return input_sample, target_label


    def __len__(self):
        return len(self.data)


class NaiveTrainer:
    def __init__(self):
        pass
    
    # # z score normalization
    # def z_score_normalization(self, inputs):
    #     mean = np.mean(inputs)
    #     std_dev = np.std(inputs)
    #     normalized_data = (inputs - mean) / std_dev
    #     return normalized_data


    # # min-max normalization
    # def min_max_normalize(self, x, inputs: pd.DataFrame):
    #     minimum = np.min(inputs, axis=0)
    #     maximum = np.max(inputs, axis=0)
    #     nor_input = (x - minimum) / (maximum - minimum)

    #     return nor_input


    # def get_normalized_input(self, inputs: pd.DataFrame):
    #     # Create a vectorized version of the function
    #     vectorized_func = np.vectorize(self.min_max_normalize)
    #     result = vectorized_func(inputs, inputs)
    #     print(result)


    def trainer(self, model, dataset_path):
        dataset = Dataset(dataset_path)
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Training
        for epoch in range(EPOCH):
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                # loss = criterion(outputs, batch_labels)
                loss = loss_function(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


if __name__ == "__main__":

    # # Set random seed for reproducibility
    # # Need to add the line torch.manual_seed(42) right before creating the model
    # torch.manual_seed(42)
    # model = NaiveNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # naive_train = NaiveTrainer()
    # # naive_train.trainer(model, "new_results/ens_train.csv")
    # naive_train.trainer(model, "new_results/ens_train_min_true.csv")
    # # Save nn model
    # model = torch.save(model.state_dict(), "nn_method/my_baby.pt")
    
    model = NaiveNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load('nn_method/my_baby.pt'))
    model.eval()

    # Test the trained model
    # test_dataset = pd.read_csv("new_results/ens_test.csv")
    # test_input = test_dataset.iloc[:, :INPUT_SIZE].values.astype(np.float32)
    # test_target = test_dataset.iloc[:, - OUTPUT_SIZE:].values.astype(np.float32)
    # test_input = torch.Tensor(test_input)
    dataset_path = "new_results/ens_test_min_true.csv"
    dataset = Dataset(dataset_path)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for inputs, labels in test_loader:
            predicted_output = model(inputs)
            # print(predicted_output)
            predicted_output = predicted_output.numpy()
            prediction = pd.DataFrame(predicted_output)
            prediction.to_csv('prediction.csv', mode='a', header=False, index=False)

    