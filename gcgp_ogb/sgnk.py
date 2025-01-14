import torch
import torch.nn as nn

class SimplifyingGraphNeuralKernel(nn.Module):
    def __init__(self, L=2 ):
        super(SimplifyingGraphNeuralKernel, self).__init__()
        # self.K = K
        self.L = L

    def updat_sigma(self, Sigma_xx, Sigma_XX, Sigma_xX):
        """
        update the sigma matrix
        """
        a_xX  = torch.clip(2 * Sigma_xX/(torch.sqrt(1 + 2*Sigma_xx)*torch.sqrt(1 + 2*Sigma_XX)),-1,1)
        a_xx  = torch.clip(2 * Sigma_xx/(torch.sqrt(1 + 2*Sigma_xx)*torch.sqrt(1 + 2*Sigma_xx)),-1,1)
        a_XX  = torch.clip(2 * Sigma_XX/(torch.sqrt(1 + 2*Sigma_XX)*torch.sqrt(1 + 2*Sigma_XX)),-1,1)

        Sigma_xX = 2/torch.pi * torch.asin(a_xX)
        Sigma_xx = 2/torch.pi * torch.asin(a_xx)
        Sigma_XX = 2/torch.pi * torch.asin(a_XX)


        return  Sigma_xx, Sigma_XX, Sigma_xX

    def nodes_gram(self, x, X, A1, A2):

        Sigma_xx    = torch.diag(torch.matmul(x, x.t())).reshape(-1,1)
        Sigma_XX    = torch.diag(torch.matmul(X, X.t())).reshape(1,-1)
        Sigma_xX  = torch.matmul(x, X.t())
        
        Sigma_xX_list = []
        Sigma_xX_list.append(Sigma_xX)
        for l in range(self.L):
            Sigma_xx, Sigma_XX, Sigma_xX =  self.updat_sigma(Sigma_xx, Sigma_XX, Sigma_xX)

            # ReLU activation
            # Sigma_xx = torch.relu(Sigma_xx)
            # Sigma_XX = torch.relu(Sigma_XX)
            # Sigma_xX = torch.relu(Sigma_xX)

            Sigma_xX_list.append(Sigma_xX)

        # nodes_gram =  torch.mean(torch.stack(Sigma_xX_list, dim=1),dim=1)
        nodes_gram = sum(Sigma_xX_list)
        return nodes_gram
