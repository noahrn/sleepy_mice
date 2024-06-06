import numpy as np 
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler

# seeding
np.random.seed(0)

# EEG features
eeg_features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta', 'rms']

def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(__file__) 
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # navigate two levels up to the project root
        config_path = os.path.join(project_root, 'config.json')
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def load_data(data_path):
    absolute_data_path = os.path.abspath(data_path)
    print(f"Loading data from {absolute_data_path}")
    data = pd.read_csv(data_path)
    return data

def load_and_process_data(normalize=True, data_path=None, lab='all'):
    if data_path is None:
        config = load_config()
        data_path = config['data_path']

    data = load_data(data_path)
    
    # add unique_id column (mouse specific ID)
    data['unique_id'] = data['lab'].astype(str) + '_' + data['mouseID'].astype(str)
    data['unique_id'] = data['unique_id'].astype('category').cat.codes + 1

    # remove lab 4
    data = data[data['lab'] != 4.0]

    # filter by lab
    if lab != 'all':
        if '+' in lab:
            labs = [float(l) for l in lab.split('+')]
            data = data[data['lab'].isin(labs)]
        else:
            data = data[data['lab'] == float(lab)]

    if normalize:
        print("Normalizing data...")
        scaler = StandardScaler()
        for mouse in data['unique_id'].unique():
            mouse_data = data[data['unique_id'] == mouse]
            for feature in eeg_features:
                data.loc[mouse_data.index, feature] = scaler.fit_transform(mouse_data[[feature]])
        df_standardized_3std = data.copy()
        
        # remove data points that are more than 3 stdv away from the mean for each EEG feature
        for mouse in df_standardized_3std['unique_id'].unique():
            mouse_data = df_standardized_3std[df_standardized_3std['unique_id'] == mouse]
            for feature in eeg_features:
                df_standardized_3std.loc[mouse_data.index, feature] = mouse_data[feature][np.abs(mouse_data[feature] - mouse_data[feature].mean()) <= 3 * mouse_data[feature].std()]
        
        print("Normalized data successfully loaded.")
        return df_standardized_3std
    else:
        print("Raw data successfully loaded.")
        return data