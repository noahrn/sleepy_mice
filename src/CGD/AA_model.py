import torch
from time import time

class AA(torch.nn.Module):
    def __init__(self, num_comp, X, model="AA", init=None, verbose=False):
        super().__init__()
        if verbose:
            print('Initializing model: ' + model)
            t1 = time()

        self.model = model

        K = num_comp
        P = X.shape[-1]
        self.X = X

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

        loss = torch.sum(torch.linalg.matrix_norm(self.X - self.X @ G_soft @ S_soft)**2)

        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
        return loss
