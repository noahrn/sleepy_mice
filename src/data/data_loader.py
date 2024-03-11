import json
import os
import pandas as pd

# open config file
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(__file__) 
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # navigate two levels up to the project root
        config_path = os.path.join(project_root, 'config.json')
    
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