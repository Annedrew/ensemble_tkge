import torch
import torch.nn as nn

class NaiveNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 72)
        self.dropout1 = nn.Dropout(0.3627584328973211)

        self.hidden2 = nn.Linear(72, 112)
        self.dropout2 = nn.Dropout(0.4647574077560369)

        self.hidden3 = nn.Linear(112, 71)
        self.dropout3 = nn.Dropout(0.250506509765534)

        self.hidden4 = nn.Linear(71, 59)
        self.dropout4 = nn.Dropout(0.22091240915822552)

        self.hidden5 = nn.Linear(59, 35)
        self.dropout5 = nn.Dropout(0.36622446084551274)

        self.hidden6 = nn.Linear(35, 47)
        self.dropout6 = nn.Dropout(0.4999958776656172)

        self.hidden7 = nn.Linear(47, 84)
        self.dropout7 = nn.Dropout(0.20299775002964626)

        self.output = nn.Linear(84, output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

        
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

        x = self.relu(self.hidden5(x))
        x = self.dropout5(x)

        x = self.relu(self.hidden6(x))
        x = self.dropout6(x)
        
        x = self.relu(self.hidden7(x))
        x = self.dropout7(x)
        x = self.output(x)
        
        return x


class AttentionNN(nn.Module):
    # Architecture of nn
    def __init__(self, input_size, hidden_size, output_size, num_heads=1):
        super(AttentionNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 75)
        self.dropout1 = nn.Dropout(0.4)
        self.hidden2 = nn.Linear(75, 90)
        self.dropout2 = nn.Dropout(0.23)
        self.hidden3 = nn.Linear(90, 60)
        self.dropout3 = nn.Dropout(0.49)
        self.hidden3 = nn.Linear(60, 88)
        self.dropout3 = nn.Dropout(0.2)
        self.attention = nn.MultiheadAttention(88, num_heads)
        self.output = nn.Linear(88, output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    # Activation function
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout3(x)
        # x = x.permute(1, 0, 2)  # Reshape for attention layer (seq_length, batch_size, embed_dim)
        x, attention_weights = self.attention(x, x, x)  # Apply attention
        # x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_length, embed_dim)
        # x = self.relu(self.output(x))
        # x = self.softmax(self.output(x))
        x = self.output(x)
        
        return x

