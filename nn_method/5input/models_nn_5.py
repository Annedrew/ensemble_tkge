import torch
import torch.nn as nn

class NaiveNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 46)
        self.dropout1 = nn.Dropout(0.15940134716405197)
        self.hidden2 = nn.Linear(46, 112)
        self.dropout2 = nn.Dropout(0.17387846155915704)
        self.hidden3 = nn.Linear(112, 59)
        self.dropout3 = nn.Dropout(0.2626507595467087)
        self.hidden4 = nn.Linear(59, 94)
        self.dropout4 = nn.Dropout(0.25178250445466166)
        self.output = nn.Linear(94, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        
    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout3(x)
        x = self.relu(self.hidden4(x))
        x = self.dropout4(x)
        x = self.output(x)
        # x = self.softmax(x)
        
        return x


class AttentionNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size, num_heads=1):
        super(AttentionNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 46)
        self.dropout1 = nn.Dropout(0.15940134716405197)
        self.hidden2 = nn.Linear(46, 112)
        self.dropout2 = nn.Dropout(0.17387846155915704)
        self.hidden3 = nn.Linear(112, 59)
        self.dropout3 = nn.Dropout(0.2626507595467087)
        self.hidden4 = nn.Linear(59, 94)
        self.dropout4 = nn.Dropout(0.25178250445466166)
        self.attention = nn.MultiheadAttention(94, num_heads)
        self.output = nn.Linear(94, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout3(x)
        x = self.relu(self.hidden4(x))
        x = self.dropout4(x)
        # x = x.permute(1, 0, 2)  # Reshape for attention layer (seq_length, batch_size, embed_dim)
        x, attention_weights = self.attention(x, x, x)  # Apply attention
        # x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_length, embed_dim)
        x = self.output(x)
        # x = self.softmax(x)
        
        return x

