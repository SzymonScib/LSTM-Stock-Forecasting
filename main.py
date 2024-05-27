import pandas as pd
import numpy as np
import torchmetrics
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

logger = TensorBoardLogger("tb_logs", name="my_model")

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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1) #Initialization of a linear layer used to reshape output from "hidden_size" dimensions to 1 dimension 

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, input):
        lstm_out, (hn, cn) = self.lstm(input)
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        loss = nn.MSELoss()(output_i, label_i)
        self.log("train_loss", loss)

        self.train_rmse(output_i, label_i)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        val_loss = nn.MSELoss()(output_i, label_i)
        self.log("val_loss", val_loss)

        self.log("validation/predictions", output_i.mean(), prog_bar=True)
        self.log("validation/labels", label_i.mean(), prog_bar=True)

        self.val_rmse(output_i, label_i)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True)

        return val_loss

# Load and process data
X, Y = data_windowing('AMD_historical_data.csv', 5)
Y = np.array(Y).reshape(-1, 1)  # Reshaping Y to a 2D array so it can be scaled by MinMaxScaler

# Normalize the data
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y = MinMaxScaler()
Y_scaled = scaler_Y.fit_transform(Y)

num_samples = X_scaled.shape[0]
train_val_size = int(0.85 * num_samples)  # 85% for training and validation
test_size = num_samples - train_val_size

# Split the data
train_val_indices, test_indices = train_test_split(np.arange(num_samples), test_size=test_size, shuffle=False, random_state=42)
train_indices, val_indices = train_test_split(train_val_indices, test_size=int(0.25 * train_val_size), random_state=42)

X_train, X_val, X_test = X_scaled[train_indices], X_scaled[val_indices], X_scaled[test_indices]
Y_train, Y_val, Y_test = Y_scaled[train_indices], Y_scaled[val_indices], Y_scaled[test_indices]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Ensure shape is (batch_size, seq_len, input_size)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)  # Ensure shape is (batch_size, seq_len, input_size)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)  # Ensure shape is (batch_size, seq_len, input_size)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# Create TensorDataset and DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize model
model = LSTMModel(input_size=1, hidden_size=5)

# Setup trainer with logger
trainer = pl.Trainer(max_epochs=500, logger=logger, log_every_n_steps=5)

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Function to generate predictions
def generate_predictions(model, dataloader):
    model.eval()  # Set model to evaluation mode
    predictions, actuals = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    return predictions, actuals

# Function to plot predictions vs actuals
def plot_predictions(predictions, actuals, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual Values')
    plt.plot(predictions, label='Predicted Values', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
