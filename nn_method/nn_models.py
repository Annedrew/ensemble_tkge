import torch
import torch.nn as nn

class NaiveNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 20)
        self.hidden2 = nn.Linear(20, 15)
        self.hidden3 = nn.Linear(15, 10)
        self.hidden4 = nn.Linear(10, 5)
        self.output = nn.Linear(5, output_size)
        self.relu = nn.ReLU()

        
    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.output(x)
        
        return x
