import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
import sys
import os.path

# 1 folder back
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# clasifier
def classifier(S_lists, labels, model_type='XGBoost', iterations=1):
    accuracies = []
    for i in tqdm(range(iterations), desc="Classifier-Loop"):
        X_train, X_test, y_train, y_test = train_test_split(S_lists, labels, test_size=0.2, random_state=iterations)
        
        if model_type == 'LGBM':
            model = LGBMClassifier(boosting_type='rf', n_estimators=100, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, verbose=-1)
        elif model_type == 'RF':
            model = RandomForestClassifier(n_estimators=100)
        elif model_type == 'XGBoost':
            model = XGBClassifier(n_estimators=100)
        else:
            raise ValueError("Invalid model type. Choose 'LGBM', 'RF', or 'XGBoost'.")
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))
    return accuracies

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 1 folder back
    matrices_dir = os.path.join(base_dir, '..', 'data/matrices')
    results_dir = os.path.join(base_dir, '..', 'results')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for subdirname in os.listdir(matrices_dir):
        if subdirname.startswith("all"):  # Skip folders that start with 'all_'
            continue

        sub_dir_path = os.path.join(matrices_dir, subdirname)
        if os.path.isdir(sub_dir_path):
            print(f"Processing: {subdirname}")

            info_files = [f for f in os.listdir(sub_dir_path) if f.startswith('info_list')]
            s_list_files = [f for f in os.listdir(sub_dir_path) if f.startswith('S_lists')]

            if info_files and s_list_files:
                info_path = os.path.join(sub_dir_path, info_files[0])
                s_list_path = os.path.join(sub_dir_path, s_list_files[0])

                with open(info_path, 'rb') as f:
                    contents = pickle.load(f)
                    X, y, y2 = contents[:3]  # Always take the first three elements

                with open(s_list_path, 'rb') as f:
                    S_lists = pickle.load(f)

                labels = y - 1  # Adjust labels to start from 0
                chosen_model = 'XGBoost'
                
                results = {}
                for K in tqdm(range(2, 11), desc="K-Loop"):
                    accuracies_list = []
                    for j in tqdm(range(5), desc="Iteration-Loop"):
                        accuracies = classifier(S_lists[K-2][j].T, labels=labels, model_type=chosen_model)
                        accuracies_list.append(accuracies)
                    
                    accuracies_list = np.array(accuracies_list).flatten()
                    results[K] = (np.mean(accuracies_list), np.std(accuracies_list, ddof=1)/np.sqrt(5))
                    print(f"For K={K}, Mean Accuracy: {results[K][0]}, Standard Deviation: {results[K][1]}")

                # Prepare results for saving in structured array format
                K_values = np.array(list(results.keys()), dtype=np.int32)
                means = np.array([results[k][0] for k in K_values], dtype=np.float32)
                std_devs = np.array([results[k][1] for k in K_values], dtype=np.float32)
                data_to_save = np.core.records.fromarrays([K_values, means, std_devs], names='K, mean, std')

                # Save the data
                result_path = os.path.join(results_dir, f"{subdirname}_accuracies.npy")
                np.save(result_path, data_to_save)
                print(f"Results saved to {result_path}")

if __name__ == '__main__':
    main()