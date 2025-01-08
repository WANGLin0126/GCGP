import torch
import math
import torch.nn as nn


class NeuralTangentKernel(nn.Module):
    def __init__(self, L=2):
        super(NeuralTangentKernel, self).__init__()
        self.L = L

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
    
    def diag(self, g):
        diag_list = []
        sigma = torch.matmul(g, g.t())
        for l in range(self.L):
            sigma, diag = self.update_diag(sigma)
            diag_list.append(diag)
        return diag_list

    def nodes_gram(self, g1, g2, A1, A2):

        sigma = torch.matmul(g1, g2.t())
        theta = sigma
        diag_list1, diag_list2 = self.diag(g1), self.diag(g2)

        for l in range(self.L):
            sigma, degree_sigma = self.update_sigma(sigma, diag_list1[l], diag_list2[l])
            theta = theta * degree_sigma + sigma

        return theta
