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
                if optimizer.param_groups[0]["lr"]<0.001:
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