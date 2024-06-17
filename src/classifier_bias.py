import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
#from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

# import s-matrices
with open('s_matrices.pkl', 'rb') as file:
    s_matrices = pickle.load(file)

# Set global parameters for plotting
plt.rcParams.update({
    'font.size': 16,
    'figure.figsize': (14, 7),
    'savefig.dpi': 300,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 10,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'errorbar.capsize': 5
})

# gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
X, y, y2 = pickle.load(open('data/15-6-24/1_a1an/info_list_20240615-220830.pkl', 'rb'))
S_lists = pickle.load(open('data/15-6-24/1_a1an/S_lists_20240615-220830.pkl', 'rb'))
C_lists = pickle.load(open('data/15-6-24/1_a1an/C_lists_20240615-220830.pkl', 'rb'))

X = np.array(X) # X-data
y = np.array(y) # sleepstage per datapoint
y2 = np.array(y2) # labs per datapoint

labels = y2 # y or y2

# define classification
def classifier(labels, model_count, model_type='LGBM'):
    accuracies = []
    for i in range(10):
        # Classification setup - RF or XGBosoter
        if model_type == 'LGBM':
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=i)
            classifier = LGBMClassifier(boosting_type='rf', n_estimators=100, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, verbose=-1)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracies.append(accuracy_score(y_test, predictions))

        if model_type == 'RF':
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=i)
            classifier = RandomForestClassifier(n_estimators=100)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracies.append(accuracy_score(y_test, predictions))

        if model_type == 'XGBoost':
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=i)
            classifier = xgb.XGBClassifier(objective="multi:softmax", num_classes=5, n_estimators=100)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracies.append(accuracy_score(y_test, predictions))

        else:
            raise ValueError("Invalid model type. Choose 'RF' or 'XGBoost'.")

    return np.mean(accuracies), np.std(accuracies), model_count

# main loop
model_count = 1  # for tracking

chosen_model = 'XGBoost' # RF or XGBooster

results = {}
for noise in noise_terms:
    for current_bias in biases:
        tensor_data = prepare_tensor(data.copy(), current_bias)
        accuracies, std_devs = [], []
        
        for K in K_values:
            mean_acc, std_acc, model_count = process_and_classify(tensor_data, labs, K, noise, model_count, model_type=chosen_model)
            accuracies.append(mean_acc)
            std_devs.append(std_acc)
        
        results[(noise, current_bias)] = (accuracies, std_devs)
        
        # new figure for each combination of noise and bias
        plt.figure()
        plt.plot(K_values, accuracies, marker='o', linestyle='-', label=f'Noise={noise}, Bias={current_bias}')
        plt.fill_between(K_values, np.array(accuracies) - np.array(std_devs), np.array(accuracies) + np.array(std_devs), alpha=0.2)
        
        # plot definitions
        plt.xlabel('Number of Components (K)')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Mean Accuracies (Noise={noise}, Bias={current_bias})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'mean_accuracy_plot_noise={noise}_bias={current_bias}.png')
        plt.close()

# save accuracies and stdv
with open('model_accuracy_statistics.txt', 'w') as file:
    file.write("Noise, Bias, K, Mean Accuracy, Standard Deviation\n")
    for key, (accs, stds) in results.items():
        for K, (acc, std) in enumerate(zip(accs, stds), start=2):
            file.write(f"{key[0]}, {key[1]}, {K}, {acc:.4f}, {std:.4f}\n")