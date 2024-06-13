import torch
from time import time

class AA(torch.nn.Module):
    def __init__(self, num_comp, X, class_weights=None, model="AA", init=None, verbose=False):
        super().__init__()
        if verbose:
            print('Initializing model: ' + model)
            t1 = time()

        self.model = model

        K = num_comp
        P = X.shape[-1]
        self.X = X
        self.class_weights = class_weights

        # Calculate class weights if not provided
        if class_weights is None:
            class_weights = torch.ones(P, dtype=torch.double)
        self.class_weights = class_weights

        self.softmaxG = torch.nn.Softmax(dim=0)
        self.softmaxS = torch.nn.Softmax(dim=-2)

        if init is None:
            self.G = torch.nn.Parameter(self.softmaxG(-torch.log(torch.rand((P, K), dtype=torch.double))))
            self.S = torch.nn.Parameter(self.softmaxS(-torch.log(torch.rand([K, P], dtype=torch.double))))
        else:
            self.G = init['G'].clone()
            self.S = init['S'].clone()

        if verbose:
            t2 = time()
            print('Model initialized in ' + str(t2 - t1) + ' seconds')


    def get_model_params(self, get_numpy=True):
        with torch.no_grad():
            S_soft = self.softmaxS(self.S)
            G_soft = self.softmaxG(self.G)
            if get_numpy:
                return G_soft.cpu().numpy(), S_soft.cpu().numpy()
            else:
                return G_soft, S_soft

    def forward(self):
        S_soft = self.softmaxS(self.S)
        G_soft = self.softmaxG(self.G)

        device = self.G.device
        self.class_weights = self.class_weights.to(device)

        residual = self.X - self.X @ G_soft @ S_soft
        #residual_squared = torch.linalg.norm(residual, ord='fro') ** 2
        #weighted_loss = torch.sum(residual_squared)

        residual_squared =  torch.sum(torch.sum((residual**2) * self.class_weights, dim=1))
        residual_normalized = torch.sum(torch.log(torch.sum((residual**2) * self.class_weights, dim=1)))
        

        if torch.isnan(residual_normalized):
            raise ValueError('Loss is NaN')
        
        return residual_normalized
