import pandas as pd
import numpy as np

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


