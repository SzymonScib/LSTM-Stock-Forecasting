import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl



def filter_data(files):
    for file in files:
        df = pd.read_csv('./data/' + file)
        df_reduced = df.iloc[750:]
        df_reduced.to_csv('./data/' + file, index=False)

def data_windowing(file, window_size):
    df = pd.read_csv('./data/' + file)
    num_records = df.shape[0]
    close_prices = df['Close'].values
    input_sequence, output = [], []
    for i in range(num_records - window_size):
        input_sequence.append(close_prices[i:i + window_size])
        output.append(close_prices[i + window_size])
    
    return np.array(input_sequence), np.array(output)


X, Y  = data_windowing('AMD_historical_data.csv', 5)

class LSTMModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        #Generating a random value for weights generated from normal distribution and setting initial biases to have a value of 0
        self.w1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.w3 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w4 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b2 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.w5 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w6 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b3 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.w7 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w8 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b4 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):

        #Math of forget gate
        percent_to_remember = torch.sigmoid((short_memory * self.w1) + (input_value * self.w2) + self.b1)
        #Math for input gate
        potential_remember_percent = torch.sigmoid((short_memory * self.w3) + (input_value * self.w4) + self.b2)
        potential_memory = torch.tanh((short_memory * self.w4) + (input_value * self.w4) + self.b3)

        updated_long_memory = ((long_memory*percent_to_remember)+(potential_remember_percent*potential_memory))
        #Math for output gate
        output_percent = torch.sigmoid((short_memory * self.w5) + (input_value * self.w6) + self.b5)
        updated_short_memory = torch.tahn(updated_long_memory) * output_percent

        return([updated_long_memory, updated_short_memory])
    
    def forward(self, input):
        long_memory = 0
        short_memory = 0
        day_values = []

        for item in input:
            day_values.append(item) 

        for i in range(input):
            long_memory, short_memory = self.lstm_unit(day_values[i], long_memory, short_memory)

        return short_memory

    def configure_optimizers(self):
        return Adam(self.parameters())
    
    def trining_steps(self, batch): #To nie wiem czy jest dobrze wsm
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2

        self.log("train_loss", loss)

        return loss
