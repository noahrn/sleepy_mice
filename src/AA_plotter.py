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
folders.sort()  # Sort the folders list

# Function to plot the last plot
def plot_last_plot(folder_path, S_lists, y2):
    K_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    acc_list_lab_xgb = []
    std_acc_list_lab_xgb = []

    y3 = y2.copy()
    y3[y2 == 5] = 4
    y3 = y3 - 1

    for i, K in enumerate(tqdm(K_list)):
        acc_list_K = []
        for j in range(5):
            S_list = [S_lists[i][j] for i in range(len(S_lists))]
            X_train, X_test, y_train, y_test = train_test_split(S_list[i].T, y3, test_size=0.2, random_state=42)
            clf = XGBClassifier(n_estimators=10, tree_method='hist', device='cuda', eval_metric='mlogloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            acc_list_K.append(acc)

        acc_list_lab_xgb.append(np.mean(acc_list_K))
        std_acc_list_lab_xgb.append(np.std(acc_list_K))

    plt.figure()
    plt.fill_between(K_list, np.array(acc_list_lab_xgb) - np.array(std_acc_list_lab_xgb),
                     np.array(acc_list_lab_xgb) + np.array(std_acc_list_lab_xgb), alpha=0.5)
    plt.plot(K_list, acc_list_lab_xgb)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of XGBoost Classifier for y3')
    plt.savefig(os.path.join(folder_path, 'xgboost_accuracy_y3.png'))
    plt.close()

# Loop through each folder
for idx, f in enumerate(folders):
    folder_path = os.path.join(path, f)
    files = os.listdir(folder_path)
    print(folder_path)
    
    # Identify the files needed
    info_file = [file for file in files if file.startswith('info_list')][0]
    s_lists_file = [file for file in files if file.startswith('S_lists')][0]
    c_lists_file = [file for file in files if file.startswith('C_lists')][0]
    
    # Load the data
    X, y, y2 = pickle.load(open(os.path.join(folder_path, info_file), 'rb'))
    S_lists = pickle.load(open(os.path.join(folder_path, s_lists_file), 'rb'))
    C_lists = pickle.load(open(os.path.join(folder_path, c_lists_file), 'rb'))

    X = np.array(X)
    y = np.array(y)
    y2 = np.array(y2)

    print(X.shape, y.shape, y2.shape, S_lists[0][0].shape, C_lists[0][0].shape)
    
    # Extract S_list and C_list
    S_list = [S_lists[i][0] for i in range(len(S_lists))]
    C_list = [C_lists[i][0] for i in range(len(C_lists))]
    K_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Plot 1: Reconstructed angles
    plt.figure()
    datashader_plot_AA_reconstructed_angles_multiple_sep(X, C_list, S_list, K_list, y - 1, plot_size=300)
    plt.title('Reconstructed Angles')
    plt.savefig(os.path.join(folder_path, 'reconstructed_angles.png'))
    plt.close()

    # Plot 2: Accuracy of XGBoost Classifier for y
    acc_list_xgb = []
    std_acc_list_xgb = []

    for i, K in enumerate(tqdm(K_list)):
        acc_list_K = []
        for j in range(5):
            S_list = [S_lists[i][j] for i in range(len(S_lists))]
            X_train, X_test, y_train, y_test = train_test_split(S_list[i].T, y, test_size=0.2, random_state=42)
            clf = XGBClassifier(n_estimators=10, tree_method='hist', device='cuda', eval_metric='mlogloss')
            clf.fit(X_train, y_train - 1)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test - 1, y_pred)
            acc_list_K.append(acc)

        acc_list_xgb.append(np.mean(acc_list_K))
        std_acc_list_xgb.append(np.std(acc_list_K) / np.sqrt(5))

    print(acc_list_xgb)
    plt.figure()
    plt.fill_between(K_list, np.array(acc_list_xgb) - np.array(std_acc_list_xgb), np.array(acc_list_xgb) + np.array(std_acc_list_xgb), alpha=0.5)
    plt.plot(K_list, acc_list_xgb)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of XGBoost Classifier for y')
    plt.savefig(os.path.join(folder_path, 'xgboost_accuracy_y.png'))
    plt.close()
    
    if idx < 4:
        # Only plot the last plot for the first 4 folders
        plot_last_plot(folder_path, S_lists, y2)

    print(f"Saved plots for folder: {f}")



