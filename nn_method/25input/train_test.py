import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from early_stopping import EarlyStopping

# # Fix the error: module not find...
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models_nn_25 import NaiveNN, AttentionNN
from constant_25 import *


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


class FindEpoch:
    def __init__(self): 
        pass


    def trainer(self, model, dataset_path):
        torch.manual_seed(42)

        train_losses = []
        eval_losses = []
        save_path = "nn_method/25input/result"
        early_stopping = EarlyStopping(save_path)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        dataset = DDataset(dataset_path)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Training
        for epoch in range(EPOCH):
            train_loss = 0
            model.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                mse = loss_function(outputs, batch_labels)
                rmse = torch.sqrt(mse)
                rmse.backward()
                optimizer.step()
                train_loss += rmse.item()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")

            # Testing
            model.eval()
            loss_function = nn.MSELoss()
            dataset = DDataset("dataset/NN/25t_5w/dataset/validation_dataset.csv")
            test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            with torch.no_grad():
                eval_loss = 0
                for inputs, labels in test_loader:
                    predicted_output = model(inputs)
                    prediction = predicted_output.numpy()
                    prediction = pd.DataFrame(prediction)

                    mse = loss_function(predicted_output, labels)
                    rmse = torch.sqrt(mse)
                    # prediction.to_csv('nn_method/25input/prediction.csv', mode='a', header=False, index=False)

                    eval_loss += rmse.item()
                    # print(f"Loss: {rmse.item()}")

            if (epoch + 1) % 1 == 0:
                print(f"Epoch: {epoch+1}, Test Loss: {eval_loss}")
                    
            if epoch > 5:
                early_stopping(train_loss, eval_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break



if __name__ == "__main__":
    naive_train = FindEpoch()
    # model = NaiveNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model = AttentionNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    naive_train.trainer(model, "dataset/NN/25t_5w/dataset/train_dataset.csv")
