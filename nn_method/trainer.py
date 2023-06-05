import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Fix the error: module not find...
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models_nn import NaiveNN, AttentionNN
from ensemble_tkge.constants import *

import torch


class DDataset(torch.utils.data.Dataset):
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


    def trainer(self, dataset_path):
        model = NaiveNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        # Set random seed for reproducibility
        # Need to add the line torch.manual_seed(42) right before creating the model
        torch.manual_seed(42)
        dataset = DDataset(dataset_path)
        loss_function = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Training
        for epoch in range(EPOCH):
            model.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                # loss = criterion(outputs, batch_labels)
                mse = loss_function(outputs, batch_labels)
                rmse = torch.sqrt(mse)
                rmse.backward()
                optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch: {epoch+1}, Loss: {rmse.item()}")

        # Save nn model
        model = torch.save(model.state_dict(), "nn_method/my_baby_naive.pt")


class AttentionTrainer:
    def __init__(self):
        pass


    def trainer(self, dataset_path):
        model = AttentionNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        # Set random seed for reproducibility
        # Need to add the line torch.manual_seed(42) right before creating the model
        torch.manual_seed(42)
        dataset = DDataset(dataset_path)
        loss_function = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Training
        for epoch in range(EPOCH):
            model.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                # loss = criterion(outputs, batch_labels)
                mse = loss_function(outputs, batch_labels)
                rmse = torch.sqrt(mse)
                rmse.backward()
                optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch: {epoch+1}, Loss: {rmse.item()}")
        
        # Save nn model
        model = torch.save(model.state_dict(), "nn_method/my_baby_attention.pt")


if __name__ == "__main__":
    naive_train = NaiveTrainer()
    naive_train.trainer("dataset/NN/25t_5w/dataset/train_dataset.csv")