import torch
import torch.nn as nn


class NaiveNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size):
        # 20, 15, 10, 5
        super(NaiveNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 5)
        self.hidden2 = nn.Linear(5, 5)
        # self.hidden3 = nn.Linear(10, 5)
        self.output = nn.Linear(5, output_size)
        self.relu = nn.ReLU()

        
    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        # x = self.relu(self.hidden3(x))
        x = self.output(x)
        
        return x


class AttentionNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size, num_heads=1):
        super(AttentionNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 5)
        self.hidden2 = nn.Linear(5, 10)
        self.attention = nn.MultiheadAttention(10, num_heads)
        self.hidden3 = nn.Linear(10, 5)
        self.output = nn.Linear(5, output_size)
        self.relu = nn.ReLU()

    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        # x = x.permute(1, 0, 2)  # Reshape for attention layer (seq_length, batch_size, embed_dim)
        x, attention_weights = self.attention(x, x, x)  # Apply attention
        # x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_length, embed_dim)
        x = self.relu(self.hidden3(x))
        x = self.output(x)
        
        return x, attention_weights

