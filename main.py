import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


def data_windowing(file, window_size):
    df = pd.read_csv('./data/' + file)
    num_records = df.shape[0]
    close_prices = df['Close'].values
    input_sequence, output = [], []
    for i in range(num_records - window_size):
        input_sequence.append(close_prices[i:i + window_size])
        output.append(close_prices[i + window_size])
    
    return np.array(input_sequence), np.array(output)


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size= hidden_size, batch_first=True)  
        self.linear = nn.Linear(hidden_size, 1)     
    
    def forward(self, input):
        
        lstm_out, temp = self.lstm(input)
        prediction = prediction = self.linear(lstm_out[:, -1, :])

        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters())
    
    def training_step(self, batch, batch_idx): 
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2

        self.log("train_loss", loss)
        self.log("out", output_i )

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        val_loss = nn.MSELoss()(output_i, label_i)
        self.log("val_loss", val_loss)

        # Log predictions for tracking
        for inp, pred, label in zip(input_i, output_i, label_i):
            self.logger.experiment.add_scalar("validation/predictions", pred.item(), self.current_epoch)
            self.logger.experiment.add_scalar("validation/labels", label.item(), self.current_epoch)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
    

X, Y  = data_windowing('AMD_historical_data.csv', 5)
Y = np.array(Y).reshape(-1, 1)#Reshaping Y to a 2D array so it can be scaled by MinMaxScaler

model = LSTMModel(1, 50)

scalerX = MinMaxScaler()
X_scaled = scalerX.fit_transform(X)
scalerY = MinMaxScaler()
Y_scaled = scalerY.fit_transform(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

trainer = pl.Trainer(max_epochs=3000)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


