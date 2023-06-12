import torch
import torch.nn as nn

class NaiveNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 72)
        self.dropout1 = nn.Dropout(0.27)
        self.output = nn.Linear(72, output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

        
    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        # x = self.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.output(x)
        
        return x


class AttentionNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size, num_heads=1):
        super(AttentionNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 30)
        self.dropout1 = nn.Dropout(0.43)
        self.hidden2 = nn.Linear(30, 17)
        self.dropout2 = nn.Dropout(0.37)
        self.attention = nn.MultiheadAttention(17, num_heads)
        self.output = nn.Linear(17, output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout2(x)
        # x = x.permute(1, 0, 2)  # Reshape for attention layer (seq_length, batch_size, embed_dim)
        x, attention_weights = self.attention(x, x, x)  # Apply attention
        # x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_length, embed_dim)
        x = self.output(x)
        
        return x

