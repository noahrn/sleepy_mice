# optimization loop
from tqdm import tqdm
import torch
import numpy as np

def Optimizationloop(model, optimizer, scheduler=None, max_iter=100, tol=1e-10,disable_output=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not disable_output:
        print("Device: ", device)

    all_loss = []
    lrs = []

    for epoch in tqdm(range(max_iter),disable=disable_output):
        loss = model()
        all_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        if epoch>5:
            if scheduler is not None:
                scheduler.step(loss)
                if optimizer.param_groups[0]["lr"]<0.0001:
                    break
            else: #specify relative tolerance threshold
                latest = np.array(all_loss[-5:])
                minval = np.min(latest)
                secondlowest = np.min(latest[latest!=minval])
                if (secondlowest-minval)/minval<tol:
                    break
                    # if (all_loss[-5]-all_loss[-1])/all_loss[-5]<tol:
                #     break
                
    if not disable_output:
        print("Tolerance reached at " + str(epoch+1) + " number of iterations")
    best_loss = min(all_loss)
    return all_loss, best_loss




def sample_data(X, y, y2, method='balanced', n_points=None):
    """
    Sample data based on the specified method.
    
    Parameters:
    - X: Input data array.
    - y: Labels array.
    - y2: Labels array for the second task, still sorted by the same indices as y.
    - method: Sampling method. One of 'balanced', 'random', or 'all'.
    - n_points: Number of points to sample if method is 'random'. Ignored for other methods.
    
    Returns:
    - X_sampled: Sampled input data.
    - y_sampled: Sampled labels.
    - y2_sampled: Sampled labels for the second task.
    """
    if method not in ['balanced', 'random', 'all']:
        raise ValueError("Invalid method. Choose from 'balanced', 'random', or 'all'.")
    
    if method == 'balanced':
        idx = []
        for i in np.unique(y):
            class_indices = np.where(y == i)[0]
            sampled_indices = np.random.choice(class_indices, 30000, replace=False)
            idx.extend(sampled_indices)
        idx = np.array(idx)
    
    elif method == 'random':
        if n_points is None:
            raise ValueError("n_points must be specified for random sampling.")
        idx = np.random.choice(len(y), n_points, replace=False)
    
    elif method == 'all':
        idx = np.arange(len(y))
    
    sorted_idx = np.sort(idx)
    X_sampled = X[:, sorted_idx]
    y_sampled = y[sorted_idx].astype(int)
    y2_sampled = y2[sorted_idx].astype(int)
    
    return X_sampled, y_sampled, y2_sampled