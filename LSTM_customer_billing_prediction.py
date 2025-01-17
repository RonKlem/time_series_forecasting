import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Variable

class LSTM(nn.Module):
    """
    LSTM model for time series forecasting.

    Attributes:
        num_classes (int): The number of output classes.
        num_layers (int): The number of recurrent layers.
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        seq_length (int): The sequence length of the time series data.
    """
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        """
        Initialize the LSTM model.

        Args:
            num_classes (int): The number of output classes.
            num_layers (int): The number of recurrent layers.
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            seq_length (int): The sequence length of the time series data.
        """

        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the LSTM layer.

        Args:
            x (torch.Tensor): The input to the LSTM layer.

        Returns:
            torch.Tensor: The output from the LSTM layer.
        """
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
