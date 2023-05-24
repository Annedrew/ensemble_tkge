import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Training
        for epoch in range(EPOCH):
            print(f"In epoch:{epoch}, total epoch:{EPOCH}")
            # dtype: torch.float64
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = loss_function(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
# 保存模型


if __name__ == "__main__":

    # Set random seed for reproducibility
    # Need to add the line torch.manual_seed(42) right before creating the model
    torch.manual_seed(42)
    model = NaiveNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    naive_train = NaiveTrainer()
    naive_train.trainer(model, "new_results/ens_train.csv")
    torch.save(model.state_dict(), "nn_method/")

    # Test the trained model
    test_dataset = pd.read_csv("new_results/ens_test.csv")
    test_input = test_dataset.iloc[:, :INPUT_SIZE].values.astype(np.float32)
    test_target = test_dataset.iloc[:, - OUTPUT_SIZE:].values.astype(np.float32)
    test_input = torch.Tensor(test_input)

    with torch.no_grad():
        predicted_output = model(test_input)
    print(predicted_output)
    # with open("temp.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(predicted_output)