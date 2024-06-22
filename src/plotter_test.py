import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.AA_visualize import datashader_plot_AA_reconstructed_angles_multiple_sep
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm

# Define the base path
path = '/work3/s224178'
folders = os.listdir(path)

folders.sort()

# Loop through each folder
for f in folders:
    folder_path = os.path.join(path, f)
    files = os.listdir(folder_path)
    
    print(folder_path)
