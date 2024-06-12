import sys
import os
import torch
import random
import numpy as np
#from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

# global parameters for pltting
mpl.rcParams.update({
    'font.size': 16,
    'figure.figsize': (14, 7),  # aspect ratio
    'savefig.dpi': 300,  # High resolution
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

# modularized data load
data = load_and_process_data(normalize=True, lab="all")
data = data.sample(frac=0.01).reset_index(drop=True)
print("Data loaded and normalized with shape:", data.shape)

# lab label to be predicted
labs = data['lab'].values

# frequency bands + rms
features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta', 'rms']
data = data[features]

# convert to tensor & use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
data_tensor = data_tensor.to(dtype=torch.float64).transpose(0, 1)
print("Transposed data tensor shape (before model):", data_tensor.shape)

def process_and_classify(labels, K):
    accuracies = []
    for _ in range(10):  # 10 iteration for each K
        model = AA(X=data_tensor, num_comp=K, model='AA', verbose=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss, _ = Optimizationloop(model=model, optimizer=optimizer, max_iter=100, tol=1e-6, disable_output=False)
        
        C, S = model.get_model_params()
        C, S = C.cpu().detach().numpy(), S.cpu().detach().numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(S.T, labels, test_size=0.2, random_state=42)
        
        # using LGBM for gpu acceleration for hpc
        classifier = LGBMClassifier(boosting_type='rf', n_estimators=100, random_state=0, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, verbose=-1)
        #classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    
    return accuracies

# Collect and plot results
K_values = range(2, 11)
mean_accuracies = []
std_accuracies = []

# Open a file to write the results
with open('model_accuracy_statistics.txt', 'w') as file:
    file.write("K, Mean Accuracy, Standard Deviation\n")

    for K in K_values:
        print(f"Running for K={K}")
        accuracies = process_and_classify(labs, K)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)
        
        # add the results to the txt
        file.write(f"{K}, {mean_acc:.4f}, {std_acc:.4f}\n")
        
        print(f"Mean Accuracy: {mean_acc}, Standard Deviation: {std_acc}")

# Plots
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.bar(K_values, mean_accuracies, yerr=std_accuracies, color='skyblue', alpha=0.8)
plt.xlabel('Number of Components (K)')
plt.ylabel('Mean Accuracy')
#plt.title('Bar Plot of Accuracy vs. K')
plt.grid(True)
plt.savefig('accuracy_bar_plot.png')

plt.subplot(1, 2, 2)
plt.plot(K_values, mean_accuracies, marker='o', linestyle='-', color='steelblue')
plt.fill_between(K_values, np.array(mean_accuracies) - np.array(std_accuracies), np.array(mean_accuracies) + np.array(std_accuracies), color='steelblue', alpha=0.2)
plt.xlabel('Number of Components (K)')
plt.ylabel('Mean Accuracy')
#plt.title('Line Plot of Mean Accuracies')
plt.grid(True)
plt.savefig('mean_accuracy_line_plot.png') 

plt.tight_layout()
plt.show()