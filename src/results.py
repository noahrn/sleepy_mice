from preprocessing.data_loader import load_and_process_data
import numpy as np
import torch
import pickle
from tqdm import tqdm
from CGD import AA_model, AA_trainer

# Load and process data
df = load_and_process_data(normalize=False, lab="1")
df['logrms'] = np.log1p(df['rms'])
df.insert(7, 'logrms', df.pop('logrms'))

# only keep ["slowdelta", "fastdelta", "slowtheta", "fasttheta", "alpha", "beta", "rms", "sleepstage"]
X = df.drop(columns=['mouseID', 'lab', 'sleepstage', 'epoch', 'unique_id','rms'])
y = df['sleepstage']
X = X.to_numpy()
y = y.to_numpy()
X = X.T

# sort by sleep stage
sort_idx = np.argsort(y)
X = X[:, sort_idx]
y = y[sort_idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.tensor(X).to(device)

# Fit the model for archetypes K
K_list = [4]

C_list_list = [[] for _ in range(len(K_list))]
S_list_list = [[] for _ in range(len(K_list))]

class_labels = torch.tensor(y, dtype=torch.long).to(device)
unique_labels = torch.unique(class_labels)

# Calculate class weights using PyTorch operations
class_weights = torch.tensor([1 / torch.sum(class_labels == i).item() for i in unique_labels]).to(device)
class_weights = class_weights / torch.sum(class_weights)

# Expand class_weights to match each sample
sample_weights = class_weights[class_labels - 1]


for i, K in tqdm(enumerate(K_list)):
    # Run each model 5 times
    for _ in range(5):
        model = AA_model.AA(X=data, num_comp=K, class_weights=sample_weights, noise_term=True, model='AA', verbose=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss,_ = AA_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=10000,tol=1e-6, disable_output=True)
        C, S = model.get_model_params()
        C_list_list[i].append(C)
        S_list_list[i].append(S)


# save the two list, they contain the C and S matrices for each K, but they have different shapes, so np.save does not work
with open('data/C_list_list.pkl', 'wb') as f:
    pickle.dump(C_list_list, f)
with open('data/S_list_list.pkl', 'wb') as f:
    pickle.dump(S_list_list, f)