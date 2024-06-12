import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler

# Seeding
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

def load_data(data_path, verbose=True):
    absolute_data_path = os.path.abspath(data_path)
    if verbose:
        print(f"Loading data from {absolute_data_path}")
    data = pd.read_csv(data_path)
    return data

def load_and_process_data(normalize=True, data_path=None, lab="all", verbose=True):
    lab = str(lab)  # convert lab to string

    if data_path is None:
        config = load_config()
        data_path = config['data_path']

    data = load_data(data_path, verbose=verbose)
    
    # remove lab 4
    data = data[data['lab'] != 4.0]

    # filter by lab
    if lab != 'all':
        labs = [float(l) for l in lab.split(',')]
        data = data[data['lab'].isin(labs)]
    else:
        labs = data['lab'].unique()

    if normalize:
        if verbose:
            if lab == 'all':
                print("Normalizing data for all labs...")
            else:
                print(f"Normalizing data for lab {lab}...")
                
        scaler = StandardScaler()
        for mouse in data['unique_id'].unique():
            mouse_data = data[data['unique_id'] == mouse]
            for feature in eeg_features:
                data.loc[mouse_data.index, feature] = scaler.fit_transform(mouse_data[[feature]])
        df_standardized_3std = data.copy()
        
        # remove data points that are more than 3 stdv away from the mean for each EEG feature
        all_outliers = []
        for mouse in df_standardized_3std['unique_id'].unique():
            mouse_data = df_standardized_3std[df_standardized_3std['unique_id'] == mouse]
            for feature in eeg_features:
                is_within_3_std = np.abs(mouse_data[feature] - mouse_data[feature].mean()) <= 3 * mouse_data[feature].std()
                outliers = mouse_data[~is_within_3_std].index
                all_outliers.extend(outliers)

        
        # Drop rows with NaN values after outlier removal
        df_standardized_3std.drop(all_outliers, inplace=True)

        if verbose:
            if lab == 'all':
                print("Normalized data successfully loaded from all labs.")
            else:
                print(f"Normalized data successfully loaded from lab {lab}.")
                
        return df_standardized_3std
    else:
        if verbose:
            print(f"Raw data successfully loaded from lab {lab}.")
        return data