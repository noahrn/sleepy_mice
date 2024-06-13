from preprocessing.data_loader import load_and_process_data
import numpy as np
import torch
import pickle
from tqdm import tqdm
from CGD import AA_model, AA_trainer

# Load and process data
df = load_and_process_data(normalize=False, lab="all")
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
K_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

C_list_list = [[] for _ in range(len(K_list))]
S_list_list = [[] for _ in range(len(K_list))]

for i, K in tqdm(enumerate(K_list)):
    # Run each model 5 times
    for _ in range(5):
        model = AA_model.AA(X=data,num_comp=K,model='AA', verbose=False)
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