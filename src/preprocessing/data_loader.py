import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler

# Seeding
#np.random.seed(0)

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

def load_and_process_data(normalize=True, data_path=None, lab="all", verbose=True, narcolepsy=False):
    lab = str(lab)  # convert lab to string
    data = None  # initialize data

    config = load_config()

    if data_path is None:
        data_path = config['data_path']

    data = load_data(data_path, verbose=verbose)

    if narcolepsy:
        narcolepsy_path = config['narcolepsy_path']
        narcolepsy_data = load_data(narcolepsy_path, verbose=verbose)
        narcolepsy_data['narcolepsy'] = 1
        narcolepsy_data.reset_index(drop=True, inplace=True)
        
        # label the healthy data as 0 for 'narcolepsy'
        data['narcolepsy'] = 0
        data.reset_index(drop=True, inplace=True)
        
        data = pd.concat([data, narcolepsy_data])
        data.reset_index(drop=True, inplace=True)
        if verbose:
            print("Loaded healthy & narcolepsy data successfully.")
    else:
        if verbose:
            print("Healthy data loaded successfully.")
    
    if verbose:
        print(data)

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
                
        # identify data points that are more than 3 stdv away from the mean for each feature
        all_outliers = []
        for mouse in data['unique_id'].unique():
            mouse_data = data[data['unique_id'] == mouse]
            for feature in eeg_features:
                is_within_3_std = np.abs(mouse_data[feature] - mouse_data[feature].mean()) <= 3 * mouse_data[feature].std()
                outliers = mouse_data[~is_within_3_std].index
                all_outliers.extend(outliers)

        # remove all observations that have outliers
        data.drop(all_outliers, inplace=True)

        # standardize for each mouse for each feature independently
        scaler = StandardScaler()
        for mouse in data['unique_id'].unique():
            mouse_data = data[data['unique_id'] == mouse]
            for feature in eeg_features:
                data.loc[mouse_data.index, feature] = scaler.fit_transform(mouse_data[[feature]])

        df_standardized_3std = data.copy()

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