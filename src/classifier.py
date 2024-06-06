import sys
import os
from CGD import AA, Optimizationloop, CGD
from preproccesing.data_loader import load_and_process_data
from sklearn import svm
import torch

data = load_and_process_data(normalize=True, lab="1")

# convert to tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_tensor = torch.tensor(data.values).to(device)

K = 5
model = AA(X=data_tensor, num_comp=K, model='AA', verbose=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss, _ = Optimizationloop(model=model, optimizer=optimizer, max_iter=10000, tol=1e-6, disable_output=False)

C, S = model.get_model_params()
C, S = C.cpu().detach().numpy(), S.cpu().detach().numpy()