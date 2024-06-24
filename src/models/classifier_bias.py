import torch
import numpy as np
import matplotlib.pyplot as plt
#from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os.path

# 1 folder back
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# modularized import
from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

# gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data manually
with open('data/matrices/3_lab_1bias_nolog/info_list_20240617-211347.pkl', 'rb') as f: # info list
    contents = pickle.load(f)
    
if len(contents) >= 1:
    X = np.array(contents[0]) # entire data
if len(contents) >= 2:
    y = np.array(contents[1]) # sleepstage per datapoint
if len(contents) >= 3:
    y2 = np.array(contents[2]) # labs per datapoint
"""
if len(contents) >= 4:
    y3 = np.array(contents[3]) # unique_id
if len(contents) >= 5:
    y4 = np.array(contents[4]) # narcolepsy
"""
    
S_lists = pickle.load(open('data/matrices/3_lab_1bias_nolog/S_lists_20240617-211347.pkl', 'rb')) # S-matrices

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

    return accuracies, y_test, predictions 

def plot_results(results):
    K = list(results.keys())
    mean_acc = [results[k][0] for k in K]
    std_acc = [results[k][1] for k in K]

    plt.plot(K, mean_acc, 'k-')  # mean curve.
    plt.fill_between(K, [m - s for m, s in zip(mean_acc, std_acc)], [m + s for m, s in zip(mean_acc, std_acc)], color='steelblue', alpha=0.5)
    plt.axhline(0.48, color='r', linestyle='--') # baseline
    plt.title(f"Model: {chosen_model}, logrms, Bias: 1")
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Accuracy")
    plt.xticks(K)
    plt.grid()
    plt.show()

# parameters
chosen_model = 'XGBoost' # LightGBM, RF or XGBoost
labels = y-1 # y-1 for sleepstages, y2 for labs, y3 for unique_id, y4 for narcolepsy for prediction
iterations = 1 # num of classifier iterations

S_lists[0][0] # first index is K, second is the iteration up to 5

# only for all labs when predicting lab else comment out
# y3 = y2.copy()
# y3[y2 == 5] = 4
# y3 = y3 - 1
# labels = y3
# print(np.unique(labels))

# main 
def main():
    results = {}
    final_confusion_matrix = None
    print("Chosen model:", chosen_model)
    for K in tqdm(range(2, 11), desc="K-Loop"):
        accuracies_list = []
        for j in tqdm(range(5), desc = "Iteration-Loop"):
            # run classifier and store predictions for potential confusion matrix plotting
            accuricies, last_y_test, last_predictions = classifier(S_lists[K-2][j].T, labels=labels, model_type=chosen_model)
            accuracies_list.append(accuricies)
            if K == 10 and j == 4:  # Check if it's the last K and last iteration
                final_confusion_matrix = confusion_matrix(last_y_test, last_predictions)
        
        # store mean and standard deviation in results dictionary for current K
        accuracies_list = np.array(accuracies_list).flatten()
        results[K] = (np.mean(accuracies_list), np.std(accuracies_list, ddof=1)/np.sqrt(5))
        print(f"For K={K}, Mean Accuracy: {results[K][0]}, Standard Deviation: {results[K][1]}")

    # Prepare results for saving
    K_values = np.array(list(results.keys()), dtype=np.int32)
    means = np.array([results[k][0] for k in K_values], dtype=np.float32)
    std_devs = np.array([results[k][1] for k in K_values], dtype=np.float32)
    data_to_save = np.core.records.fromarrays([K_values, means, std_devs], names='K, mean, std')

    # Save the data
    np.save('', data_to_save)

    # plot the confusion matrix for the last K if it exists
    if final_confusion_matrix is not None:
        plt.figure(figsize=(5, 3))
        label_values = ['1, REM', '2, Wake', '3, NREM'] # for sleepstage
        #label_values = ['Lab 1', 'Lab 2', 'Lab 3', 'Lab 5'] # for lab
        #label_values = ['0, Healthy', '1, Nacroleptic']
        disp = ConfusionMatrixDisplay(confusion_matrix=final_confusion_matrix, display_labels=label_values)
        disp.plot(cmap='viridis', values_format='d')
        plt.title(f"Confusion Matrix for K=10, sleepstage, lab 1, log loss")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        np.save('results/cm/cm_ss_SSE_3lab.npy', final_confusion_matrix)

    # plot
    plot_results(results)

if __name__ == '__main__':
    main()