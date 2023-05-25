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


if __name__ == "__main__":
    model = NaiveNN()
    model.load_state_dict(torch.load('nn_method/my_baby.pt'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Test accuracy: {}%'.format(accuracy))


    # model = model["model"]
    # # Test the trained model
    # test_dataset = pd.read_csv("new_results/ens_test.csv")
    # test_input = test_dataset.iloc[:, :INPUT_SIZE].values.astype(np.float32)
    # test_target = test_dataset.iloc[:, - OUTPUT_SIZE:].values.astype(np.float32)
    # test_input = torch.Tensor(test_input)

    # with torch.no_grad():
    #     predicted_output = model(test_input)

    # # Save predictions
    # predicted_output = predicted_output.numpy()
    # prediction = pd.DataFrame(predicted_output)
    # prediction.to_csv('prediction.csv', index=False)