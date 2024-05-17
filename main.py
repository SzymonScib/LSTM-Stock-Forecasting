from scraper import scrape_historical_data, headers
import os
import pandas as pd

def filter_data(files):
    for file in files:
        df = pd.read_csv('./data/' + file)
        df_reduced = df.iloc[750:]
        df_reduced.to_csv('./data/' + file, index=False)

def sliding_window(file):
    df = pd.read_csv('./data/' + file)
    num_records = df.shape[0]
    means = []

    for i in range(int(num_records/5)):
        subset = df['Close'].iloc[i:i+5]
        mean_price = subset.mean()
        means.append(mean_price)