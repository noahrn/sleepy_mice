import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

# Load data
data = load_and_process_data(normalize=False, lab="all")

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

# Setup device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define features and prepare tensor
features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta']
labs = data['lab'].values

# Prepare data tensor
def prepare_tensor(data, bias):
    data['logrms'] = np.log1p(data['rms']) / bias
    feature_data = data[features + ['logrms']]
    tensor = torch.tensor(feature_data.values, dtype=torch.float32).to(device)
    return tensor.to(dtype=torch.float64).transpose(0, 1)

# Define the processing and classification function
def process_and_classify(tensor_data, labels, K, noise, model_count):
    accuracies = []
    for i in range(10):
        print(f"Model {model_count} of 540: Noise={noise}, Bias={current_bias}, K={K}, Run={i+1}/10")
        model_count += 1
        model = AA(X=tensor_data, num_comp=K, noise_term=noise, model='AA', verbose=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss, _ = Optimizationloop(model=model, optimizer=optimizer, max_iter=10000, tol=1e-6, disable_output=True)
        
        C, S = model.get_model_params()
        C, S = C.cpu().detach().numpy(), S.cpu().detach().numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(S.T, labels, test_size=0.2)
        classifier = LGBMClassifier(boosting_type='rf', n_estimators=100, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, verbose=-1)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))
    return np.mean(accuracies), np.std(accuracies), model_count

# Main loops
noise_terms = [True, False]
biases = [1, 3, 6]
K_values = range(2, 11)
model_count = 1  # Initialize model count

results = {}
for noise in noise_terms:
    for current_bias in biases:
        tensor_data = prepare_tensor(data.copy(), current_bias)
        accuracies, std_devs = [], []
        
        for K in K_values:
            mean_acc, std_acc, model_count = process_and_classify(tensor_data, labs, K, noise, model_count)
            accuracies.append(mean_acc)
            std_devs.append(std_acc)
        
        results[(noise, current_bias)] = (accuracies, std_devs)
        
        # Create a new figure for each combination of noise and bias
        plt.figure()
        plt.plot(K_values, accuracies, marker='o', linestyle='-', label=f'Noise={noise}, Bias={current_bias}')
        plt.fill_between(K_values, np.array(accuracies) - np.array(std_devs), np.array(accuracies) + np.array(std_devs), alpha=0.2)
        
        # Add labels, title, grid, and legend for each plot
        plt.xlabel('Number of Components (K)')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Mean Accuracies (Noise={noise}, Bias={current_bias})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'mean_accuracy_plot_noise={noise}_bias={current_bias}.png')
        plt.close()

# Write results to a file
with open('model_accuracy_statistics.txt', 'w') as file:
    file.write("Noise, Bias, K, Mean Accuracy, Standard Deviation\n")
    for key, (accs, stds) in results.items():
        for K, (acc, std) in enumerate(zip(accs, stds), start=2):
            file.write(f"{key[0]}, {key[1]}, {K}, {acc:.4f}, {std:.4f}\n")