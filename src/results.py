from preprocessing.data_loader import load_and_process_data
import numpy as np
import torch
import pickle
from tqdm import tqdm
from CGD import AA_model, AA_trainer
import pandas as pd
import matplotlib.pyplot as plt
import time

# wider pd print
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load and process data
df = load_and_process_data(normalize=False, lab="all")
df['logrms'] = np.log1p(df['rms'])
df.insert(7, 'logrms', df.pop('logrms'))


# only keep ["slowdelta", "fastdelta", "slowtheta", "fasttheta", "alpha", "beta", "logrms"]
df = df.sort_values("sleepstage")
X = df[["slowdelta", "fastdelta", "slowtheta", "fasttheta", "alpha", "beta", "logrms"]]
y = df['sleepstage']
y2 = df['lab'].to_numpy()


print("Added logrms and removed non-feature columns")
print(X)


X = X.to_numpy()
y = y.to_numpy()
X = X.T


X, y, y2 = AA_trainer.sample_data(X, y, y2, method='random', n_points=90_000)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.tensor(X).to(device)

print(f"Device: {device}")


# Fit the model for archetypes K
K_list = [4, 5]


C_lists = [[] for _ in range(len(K_list))]
S_lists = [[] for _ in range(len(K_list))]


class_labels = torch.tensor(y, dtype=torch.long).to(device)
unique_labels = torch.unique(class_labels)

# Calculate class weights using PyTorch operations
class_weights = torch.tensor([1 / torch.sum(class_labels == i).item() for i in unique_labels]).to(device)
class_weights = class_weights / torch.sum(class_weights) * 10

# Expand class_weights to match each sample
sample_weights = class_weights[class_labels - 1]



for i, K in enumerate(K_list):
    # Run each model 5 times
    for j in range(5):
        print((i, j))
        model = AA_model.AA(X=data, num_comp=K, class_weights=sample_weights, noise_term=False, model='AA', verbose=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        loss,_ = AA_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=1000,tol=1e-6, disable_output=True)
        C, S = model.get_model_params()
        C_lists[i].append(C)
        S_lists[i].append(S)



# save the two list with pickle
with open(f'data/C_lists{time}.pkl', 'wb') as f:
    pickle.dump(C_lists, f)

with open(f'data/S_lists{time}.pkl', 'wb') as f:
    pickle.dump(S_lists, f)


info_list = [[] for _ in range(3)]

# Save, X, y and y2
info_list[0] = X
info_list[1] = y
info_list[2] = y2

# Save the extra list, name it the current time
time = time.strftime("%Y%m%d-%H%M%S")
with open(f'data/info_list_{time}.pkl', 'wb') as f:
    pickle.dump(info_list, f)