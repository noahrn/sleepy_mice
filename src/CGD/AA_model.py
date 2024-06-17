import torch
from time import time

class AA(torch.nn.Module):
    def __init__(self, num_comp, X, class_weights=None, noise_term=True, model="AA", init=None, verbose=False):
        super().__init__()
        if verbose:
            print('Initializing model: ' + model)
            t1 = time()

        self.model = model
        self.noise_term = noise_term

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
        
        #print(torch.sum((residual**2), dim=1))
        #print(torch.sum((residual**2) * self.class_weights, dim=1))

        if self.noise_term:
            loss = torch.sum(torch.log(torch.sum((residual**2) * self.class_weights + 1e-10, dim=1)))
        else:
            loss = torch.sum((residual**2) * self.class_weights)

            
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
        
<<<<<<< HEAD
        return loss
=======
        return loss
>>>>>>> new_classifiers
