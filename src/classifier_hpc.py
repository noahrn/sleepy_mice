import sys
import os
import torch
import random
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from CGD import AA, Optimizationloop
from preprocessing.data_loader import load_and_process_data

"""
Data loading & preprocessing
"""
# modularized data load
data = load_and_process_data(normalize=True, lab="all")
data = data.sample(frac=1, random_state=0).reset_index(drop=True) # test with different subset fractions
print("Data loaded and normalized with shape:", data.shape)

# labels to be classified
labs = data['lab'].values
mice = data['unique_id'].values

# use only the frequency bands
features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta', 'rms']
data = data[features]

# convert to tensor & use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
data_tensor = data_tensor.to(dtype=torch.float64).transpose(0, 1)
print("Transposed data tensor shape (before model):", data_tensor.shape)

# Function to process data, train model and classify
def process_and_classify(labels, K):
    print(f"Processing for K={K}")
    # model setup (AA) & optimizationloop
    model = AA(X=data_tensor, num_comp=K, model='AA', verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss, _ = Optimizationloop(model=model, optimizer=optimizer, max_iter=100000, tol=1e-6, disable_output=True)
    
    C, S = model.get_model_params()
    C, S = C.cpu().detach().numpy(), S.cpu().detach().numpy()
    print("Matrix C shape:", C.shape, "Matrix S shape:", S.shape)

    # split
    X_train, X_test, y_train, y_test = train_test_split(S.T, labels, test_size=0.2, random_state=42)
    
    # randomforest from lightgbm
    classifier = LGBMClassifier(boosting_type='rf', n_estimators=100, random_state=0, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, verbose=-1)
    
    # train
    print("Training classifier...")
    classifier.fit(X_train, y_train)
    
    # test
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy of classifier on test set:", accuracy)
    #print("Classification Report:\n", classification_report(y_test, predictions))
    
    # check if data is label-driven
    if accuracy < 0.5: 
        print("Data appears not to be driven by the label.")
    else:
        print("Data may contain patterns driven by the label.")
    print("\n" + "-"*75 + "\n")

# Loop over K values from 2 to 10
for K in range(2, 10):
    print(f"\n" + "-"*75 + f"\nClassifying based on 'lab' labels with K={K}\n" + "-"*75)
    process_and_classify(labs, K)

"""
for K in range(2, 11):
    print(f"\n" + "-"*75 + f"\nClassifying based on 'unique_id' labels with K={K}\n" + "-"*75)
    process_and_classify(mice, K)
"""