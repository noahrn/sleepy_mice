import torch
import numpy as np
import matplotlib.pyplot as plt
#from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm

# modularized import
from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

# gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
X, y, y2 = pickle.load(open('data/matrices/15-6-24/1_a1an/info_list_20240615-220830.pkl', 'rb')) # info
S_lists = pickle.load(open('data/matrices/15-6-24/1_a1an/S_lists_20240615-220830.pkl', 'rb')) # S-matrices
# C_lists = pickle.load(open('data/matrices/15-6-24/1_a1an/C_lists_20240615-220830.pkl', 'rb')) # C-matrices

X = np.array(X) # entire data
y = np.array(y) # sleepstage per datapoint
y2 = np.array(y2) # labs per datapoint
print("Data loaded with shape:", y2.shape)

# define classification
def classifier(S_lists, labels, model_type='LGBM'):
    accuracies = []
    for i in tqdm(range(iterations), desc="Classifier-Loop"):
    #for i in range(iterations):
        # Classification setup 
        X_train, X_test, y_train, y_test = train_test_split(S_lists, labels, test_size=0.2, random_state=iterations)

        if model_type == 'LGBM':
            classifier = LGBMClassifier(boosting_type='rf', n_estimators=100, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, verbose=-1)
        
        elif model_type == 'RF':
            classifier = RandomForestClassifier(n_estimators=100)
        
        elif model_type == 'XGBoost':
            classifier = xgb.XGBClassifier(n_estimators=100)
        
        else:
            raise ValueError("Invalid model type. Choose 'LGBM', 'RF' or 'XGBoost'.")

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    return accuracies

# set global parameters for plotting
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

def plot_results(results):
    K = list(results.keys())
    mean_acc = [results[k][0] for k in K]
    std_acc = [results[k][1] for k in K]

    plt.plot(K, mean_acc, 'k-')  # mean curve.
    plt.fill_between(K, [m - s for m, s in zip(mean_acc, std_acc)], [m + s for m, s in zip(mean_acc, std_acc)], color='steelblue', alpha=0.5)
    plt.axhline(0.48, color='r', linestyle='--') # baseline
    plt.title(f"Model: {chosen_model}")
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Accuracy")
    plt.xticks(K)
    plt.grid()
    plt.show()

results = {}

# parameters
chosen_model = 'XGBoost' # LightGBM, RF or XGBooster
labels = y2 # y for sleepstages or y2 for labs
iterations = 5 # num of classifier iterations

S_lists[0][0] # first index is K, second is the iteration up to 5

y3 = y2.copy()
y3[y2 == 5] = 4
y3 = y3 - 1

labels = y3

# main 
def main():
    results = {}     
    print("Chosen model:", chosen_model)
    for K in tqdm(range(2, 11), desc="K-Loop"):
        accuracies_list = []
        for j in tqdm(range(5), desc = "Iteration-Loop"):
            accuricies = classifier(S_lists[K-2][j].T, labels=labels, model_type=chosen_model)
            accuracies_list.append(accuricies)

        # store mean and standard deviation in results dictionary for current K
        accuracies_list = np.array(accuracies_list).flatten()
        print(accuracies_list)
        results[K] = (np.mean(accuracies_list), np.std(accuracies_list))
        print(f"For K={K}, Mean Accuracy: {results[K][0]}, Standard Deviation: {results[K][1]}")

    # plot
    plot_results(results)

if __name__ == '__main__':
    main()