import sys
import os
import torch
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

"""
Data loading & preprocessing
"""
# modularized data load
data = load_and_process_data(normalize=True, lab="1")
data = data.sample(frac=1, random_state=0).reset_index(drop=True) # test with different subset fractions
print("Data loaded and normalized with shape:", data.shape)

# check NAN
#print("Number of NaN values in data:", data.isnull().sum().sum())

# labels to be classified
labels = ['lab', 'unique_id']

# use only the frequency bands
features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta', 'rms']
data = data[features]

# convert to tensor & use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
data_tensor = data_tensor.to(dtype=torch.float64).transpose(0, 1)
print("Transposed data tensor shape (before model):", data_tensor.shape)

# artchetypal components
K_values = range(2, 11)

for label in labels:
    for K in K_values:
        # model setup (AA) & optimizationloop
        model = AA(X=data_tensor, num_comp=K, model='AA', verbose=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss, _ = Optimizationloop(model=model, optimizer=optimizer, max_iter=10000, tol=1e-6, disable_output=False)

        C, S = model.get_model_params()
        C, S = C.cpu().detach().numpy(), S.cpu().detach().numpy()
        #print("Matrix C shape:", C.shape, "Matrix S shape:", S.shape)

        """
        Classifiers
        """
        # split
        X_train, X_test, y_train, y_test = train_test_split(S.T, data[label].values, test_size=0.2, random_state=42)

        # randomforest from sklearn
        classifier = RandomForestClassifier(n_estimators=100, random_state=0)

        # train
        print(f"Training {label} classifier with K={K}...")
        classifier.fit(X_train, y_train)

        # test
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy of classifier on test set:", accuracy)
        print("Classification Report:\n", classification_report(y_test, predictions))

        # check if data is driven by current label
        if accuracy < 0.5: 
            print(f"Data appears not to be {label}-driven.")
        else:
            print(f"Data may contain {label}-driven patterns.")

        print("\n" + "-"*75 + "\n")