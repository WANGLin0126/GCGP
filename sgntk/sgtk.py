import torch
import math
import torch.nn as nn



class SimplifyingGraphTangentKernel(nn.Module):
    def __init__(self,K=2, L=2):
        super(SimplifyingGraphTangentKernel, self).__init__()
        self.K = K
        self.L = L
        # self.scale  = scale

    def aggr(self, S, A1, A2):
        """
        Aggregation opteration on sparse or dense matrix
        S = A1 * S * A2.t()
        """
        if A1.is_sparse and A2.is_sparse:                   # A1, A2 are sparse
            S = torch.sparse.mm(A1,S)
            S = torch.sparse.mm(A2,S.t()).t()
        elif A1.is_sparse and not A2.is_sparse:             # A1 is sparse, A2 is dense
            S = torch.sparse.mm(A1,S)
            S = torch.matmul(S,A2.t())
        elif not A1.is_sparse and A2.is_sparse:             # A1 is dense, A2 is sparse
            S = torch.matmul(A1,S)
            S = torch.sparse.mm(A2,S.t()).t()
        else:                                               # A1, A2 are dense
            S = torch.matmul(torch.matmul(A1,S),A2.t())     # (A1 * S) * A2.t()
        return S

    def update_sigma(self, S, diag1, diag2):
        S    = S / diag1[:, None] / diag2[None, :]
        S    = torch.clip(S,-0.9999,0.9999)
        S    = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        degree_sigma   = (math.pi - torch.arccos(S)) / math.pi
        S    = S * diag1[:, None] * diag2[None, :]
        return S, degree_sigma
    
    def update_diag(self, S):
        diag = torch.sqrt(torch.diag(S))
        S    = S / diag[:, None] / diag[None, :]
        S    = torch.clip(S,-0.9999,0.9999)
        S  = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        S    = S * diag[:, None] * diag[None, :]
        return S, diag
    
    def diag(self, g, A):
        diag_list = []
        sigma = torch.matmul(g, g.t())
        # for k in range(self.K):
        #     sigma = self.aggr(sigma, A, A)
        for l in range(self.L):
            sigma, diag = self.update_diag(sigma)
            diag_list.append(diag)
        return diag_list

    def nodes_gram(self, g1, g2, A1, A2):   

        sigma = torch.matmul(g1, g2.t())
        theta = sigma
        diag_list1, diag_list2 = self.diag(g1, A1), self.diag(g2, A2)

        # for k in range(self.K):
        #     sigma = self.aggr(sigma, A1, A2)
        #     theta = self.aggr(theta, A1, A2)

        for l in range(self.L):
            sigma, degree_sigma = self.update_sigma(sigma, diag_list1[l], diag_list2[l])
            theta = theta * degree_sigma + sigma

        return theta
