import json
import os
import pandas as pd

# open config file
def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

if __name__ == '__main__':
    config = load_config()

    data_path = config['data_path']

    data = load_data(data_path)