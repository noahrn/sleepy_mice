# Torch file specifying trimodal, multisubject AA
import torch
from time import time

class CGD(torch.nn.Module):
    """
    Coupled generator decomposition model for variable number of input data dimensions.

    ***The input data X should always be a dictionary of torch tensors. ***

    If you have data of varying dimensions, such as EEG/MEG, then the data for these should
    be in each their own tensor, and the tensors should be input as a dictionary, e.g., X = {'EEG':EEGdata,'MEG':MEGdata}.
    For example, EEGdata could be of size (*,N_EEG,P):
    *       - indicates other dimensions, e.g., subjects or conditions, across which N and P are equal.
    N_EEG   - the number of EEG sensors (this may vary between dictionary entries)
    P       - the number of samples (this is assumed equal across dictionary entries)

    Required inputs:
    num_comp - the number of components to learn
    X - a dictionary of torch tensors.

    Optional inputs:
    model - the model to use. Options are 'SPCA', 'AA', 'DAA'. Default is 'SPCA'.
    Xtilde - a dictionary of torch tensors. If not specified, Xtilde is assumed to be equal to X.
    G_idx - a boolean tensor of size (P,) indicating which dimensions to construct Xtilde from (default all ones).
    lambda1 - the L1 regularization parameter for SPCA. 
    lambda2 - the L2 regularization parameter for SPCA.
    init - a dictionary of torch tensors, specifying the initial values for G (and S, if model=='AA').

    The models that may be learned using coupled generator decomposition are:
    SpPCA: Sparse Principal Component Analysis, in which l1 and l2 regularization coefficients must be provided.
    For sparse PCA, the model optimizes a shared generator matrix G (P,K) and a mixing matrix S (*,K,P) is
    inferred using a procrustes transformation through (X.T@Xtilde)@G. 
    AA: Archetypal Analysis, in which the model optimizes a shared generator matrix G (P,K) and a mixing matrix S (*,K,P).
    Both G and S are assumed non-negative and sum-to-one constraints enforced through the softmax function. 
    DAA: Directional Archetypal Analysis, which works similarly to AA except the data are assumes to be on a sign-invariant hypersphere. 

    Author: Anders S Olsen, DTU Compute, 2023-2024
    Latest update February 2024
    """

    def __init__(self,  num_comp, X, Xtilde=None, model="SPCA", G_idx=None,lambda1=None,lambda2=None,init=None, verbose=False):
        super().__init__()
        if verbose:
            print('Initializing model: '+model)
        t1 = time()
        
        self.model = model
        self.keys = X.keys()

        num_modalities = len(X) #infer number of modalities from number of entries in dictionary X
        P = X[list(self.keys)[0]].shape[-1] #P should be equal across all dictionary entries
        other_dims = list(X[list(self.keys)[0]].shape[:-2]) #other dimensions, except N

        # Allow for the shared generator matrix to only learn from part of the data (dimension P),
        # such as the post-stimulus period in evoked-responses, while S covers the whole signal
        if G_idx is None:
            G_idx = torch.ones(P,dtype=torch.bool)
        self.G_idx = G_idx
        
        self.X = X
        if Xtilde is None:
            self.Xtilde = {}
            for key in self.keys:
                self.Xtilde[key] = X[key][..., self.G_idx].clone()
        
        # precompute some quantities for the SPCA and AA models
        if model == "SPCA" or model == "AA":
            self.Xsqnorm = torch.zeros(num_modalities,dtype=torch.double)
            self.XtXtilde = torch.zeros((num_modalities,*other_dims,P,torch.sum(G_idx)),dtype=torch.double)
            for m,key in enumerate(self.keys):
                self.Xsqnorm[m] = torch.sum(torch.linalg.matrix_norm(self.X[key],ord='fro')**2)
                self.XtXtilde[m] = torch.transpose(self.X[key], -2, -1) @ self.Xtilde[key]

        self.S_size = [num_modalities, *other_dims, num_comp, P]
        
        # initialize variables
        if self.model=='AA' or self.model=='DAA':
            self.softmaxG = torch.nn.Softmax(dim=0)
            self.softmaxS = torch.nn.Softmax(dim=-2)

            if init is None:
                self.G = torch.nn.Parameter(
                    self.softmaxG(-torch.log(torch.rand((torch.sum(G_idx).int(), num_comp), dtype=torch.double)))
                )
                self.S = torch.nn.Parameter(
                    self.softmaxS(-torch.log(torch.rand(self.S_size, dtype=torch.double)))
                )
            else:
                self.G = init['G'].clone()
                self.S = init['S'].clone()
        elif self.model=='SPCA':
            if lambda1 is None or lambda2 is None:
                raise ValueError('lambda1 and lambda2 must be specified for SPCA')
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.softplus = torch.nn.Softplus()
            if init is None:
                self.Bp = torch.nn.Parameter(torch.rand((torch.sum(G_idx).int(), num_comp), dtype=torch.double))
                self.Bn = torch.nn.Parameter(torch.rand((torch.sum(G_idx).int(), num_comp), dtype=torch.double))
            else:
                self.Bp = torch.nn.Parameter(init['Bp'].clone())
                self.Bn = torch.nn.Parameter(init['Bn'].clone())
        t2 = time()
        if verbose:
            print('Model initialized in '+str(t2-t1)+' seconds')

    def get_model_params(self):
        with torch.no_grad():
            if self.model == "AA" or self.model=="DAA":
                return self.softmaxG(self.G).detach(), self.softmaxS(self.S).detach()
            elif self.model == "SPCA":
                Bpsoft = self.softplus(self.Bp)
                Bnsoft = self.softplus(self.Bn)
                G = Bpsoft - Bnsoft 
                S = torch.zeros(self.S_size,dtype=torch.double)
                U,_,Vt = torch.linalg.svd(self.XtXtilde @ G,full_matrices=False)
                S = torch.transpose(U@Vt,-2,-1)
                return G, S,self.Bp.detach(),self.Bn.detach()

    def eval_model(self, Xtrain,Xtest,Xtraintilde=None,G_idx=None):
        with torch.no_grad():
            if G_idx is None:
                G_idx = torch.ones(Xtrain[list(self.keys)[0]].shape[-1],dtype=torch.bool)
            if Xtraintilde is None:
                Xtraintilde = {}
                for key in self.keys:
                    Xtraintilde[key] = Xtrain[key][..., G_idx].clone()
            if self.model == 'AA' or self.model == 'DAA':
                S = self.softmaxS(self.S)
                G = self.softmaxG(self.G)
            elif self.model=='SPCA':
                Bpsoft = self.softplus(self.Bp)
                Bnsoft = self.softplus(self.Bn)
                G = Bpsoft - Bnsoft 
                num_modalities = len(Xtrain)
                P = Xtrain[list(self.keys)[0]].shape[-1]
                other_dims = list(Xtrain[list(self.keys)[0]].shape[:-2])
                XtXtilde = torch.zeros((num_modalities,*other_dims,P,torch.sum(G_idx)),dtype=torch.double)
                for m,key in enumerate(self.keys):
                    XtXtilde[m] = torch.transpose(Xtrain[key], -2, -1) @ Xtraintilde[key]
                U,_,Vt = torch.linalg.svd(XtXtilde@G,full_matrices=False)
                S = torch.transpose(U@Vt,-2,-1)
            
            loss = 0
            for m,key in enumerate(self.keys):
                loss += torch.sum(torch.linalg.matrix_norm(Xtest[key]-Xtraintilde[key]@G@S[m])**2)
        return loss.item()

    def forwardDAA(self, X, Xtilde,G_soft,S_soft):
        loss = 0 
        for key in self.keys:
            XG = Xtilde[key] @ G_soft
            XGtXG = torch.swapaxes(XG, -2, -1) @ XG
            XGtXG = torch.swapaxes(XG, -2, -1) @ XG
            XtXG = torch.swapaxes(X[key], -2, -1) @ XG

            q = torch.sum(XGtXG @ S_soft * S_soft, dim=-2)
            z = torch.sum(torch.swapaxes(XtXG, -2, -1) * S_soft, dim=-2)
            v = (1 / torch.sqrt(q)) * z

            loss += -torch.sum(v**2)
        return loss
    
    def forwardAA(self,G_soft,S_soft):
        loss = 0
        for m,key in enumerate(self.keys):
            loss += torch.sum(torch.linalg.matrix_norm(self.X[key]-self.Xtilde[key]@G_soft@S_soft[m])**2)

        return loss
    
    def forwardSPCA(self,G):
        loss = 0
        XtXG = self.XtXtilde @ G
        U,_,Vt = torch.linalg.svd(XtXG,full_matrices=False)
        S = torch.transpose(U@Vt,-2,-1)
        loss+= torch.sum(self.Xsqnorm) - 2 * torch.sum(torch.transpose(XtXG, -2, -1) * S)
        for key in self.keys:
            XG = self.Xtilde[key] @ G
            SSE = torch.sum(XG*XG)
            loss += SSE
        return loss

    def forward(self):

        if self.model == 'AA' or self.model == 'DAA':
            S_soft = self.softmaxS(self.S)
            G_soft = self.softmaxG(self.G)
        elif self.model=='SPCA':
            Bpsoft = self.softplus(self.Bp)
            Bnsoft = self.softplus(self.Bn)
            G = Bpsoft - Bnsoft 
        
        # loop through modalities   
        if self.model == "AA":
            loss = self.forwardAA(G_soft,S_soft)
        elif self.model == "DAA":
            loss = self.forwardDAA(self.X, self.Xtilde,G_soft,S_soft)
        elif self.model == 'SPCA':
            loss = self.forwardSPCA(G)
            loss+=self.lambda1*torch.sum((Bpsoft+Bnsoft))
            loss+=self.lambda2*torch.sum((Bpsoft**2+Bnsoft**2))

        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
        return loss