import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel, ridge, K=2):
        super(KernelRidgeRegression, self).__init__()
        self.kernel   = kernel
        self.ridge    = ridge
        self.K        = K

    def discretize(self, Alpha, train=True,  epoch=0):
        if train:
            """ Gumbel sigmoid sampling """
            n, m = Alpha.shape
            delta = torch.zeros(n,m).to(Alpha.device)
            vals = torch.rand(n * (m+1) // 2).to(Alpha.device)
            i, j = torch.triu_indices(n, m)
            delta[i, j] = vals
            delta.transpose(0,1)[i, j] = vals

            """ an annealing schedule for the temperature """
            temp_start = 0.3
            temp_end = 0.01
            t = max(temp_start*(temp_end/temp_start)**(epoch/100), temp_end)

            # delta = -torch.log( - torch.log(delta+ 1e-8) + 1e-8)           # Gumbel sampling
            # adj = F.softmax((torch.log(Alpha+ 1e-8) + delta)/t, dim=1)       # Gumbel softmax


            # Alpha = (torch.log(delta) - torch.log(1-delta) + Alpha)
            # adj = torch.log(delta) - torch.log(1-delta) + adj
            
            Alpha = torch.clamp(Alpha, min=0) + 1e-8

            adj = torch.sigmoid((torch.log(delta) - torch.log(1-delta) + torch.log(Alpha))/t)
            adj = torch.clamp(adj+torch.eye(adj.shape[0]).to(adj.device), min=0, max=1)

        
        else:
            """ Using the threshold 0.5 to discretize the adjacency matrix """
            # adj = torch.sigmoid(Alpha)
            adj = Alpha/(1+Alpha)
            # TODO: 需要优化的参数和 adj 是分开的， 需要优化的参数是 Alpha ， adj 是通过 Alpha 计算出来的
            # A = adj.detach()
            # A = adj/(1+adj)
            threshold = 0.5
            adj[adj> threshold] = 1
            adj[adj<= threshold] = 0
            adj = torch.clamp(adj+torch.eye(adj.shape[0]).to(adj.device), min=0, max=1)
        
        return adj
    

    def GCF(self, adj, x, k=3):
        """
        Graph convolution filter
        parameters:
            adj: torch.Tensor, adjacency matrix, must be self-looped
            x: torch.Tensor, features
            k: int, number of hops
        return:
            torch.Tensor, filtered features
        """
        D = torch.sum(adj,dim=1)
        D = torch.pow(D,-0.5)
        D = torch.diag(D)
        
        filter = torch.matmul(torch.matmul(D,adj),D)
        for i in range(k):
            x = torch.matmul(filter,x)
        return x


    def forward(self, G_t, G_s, y_t, y_s, A_t, A_s, Alpha, train, epoch=0,  learnA=0, norm = 0):
        if learnA:
            if train:
                A_s = self.discretize( Alpha, train, epoch=epoch )
            else:
                A_s = self.discretize( Alpha, train )
                # print(sum(A_s))
        if norm:
            G_s = F.normalize(G_s, dim=1)

        G_s = self.GCF(A_s, G_s, self.K)
        # print(sum(A_s))

        K_ss      = self.kernel(G_s, G_s, A_s, A_s)
        K_ts      = self.kernel(G_t, G_s, A_t, A_s)
        n        = torch.tensor(len(G_s), device = G_s.device)
        regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device = G_s.device) / n
        b         = torch.linalg.solve(K_ss + regulizer, y_s)
        pred      = torch.matmul(K_ts, b)

        pred      = nn.functional.softmax(pred, dim = 1)
        correct   = torch.eq(pred.argmax(1).to(torch.float32), y_t.argmax(1).to(torch.float32)).sum().item()
        acc       = correct / len(y_t)
        # MSE loss for the sparse matrix A_s
        loss_sparse = torch.square(torch.sigmoid(A_s).mean() - 5/70)
        return pred, correct,loss_sparse
    

    # def GCF(self, adj, x, k=3):
    #     """
    #     Graph convolution filter
    #     parameters:
    #         adj: torch.Tensor, adjacency matrix, must be self-looped
    #         x: torch.Tensor, features
    #         k: int, number of hops
    #     return:
    #         torch.Tensor, filtered features
    #     """
    #     D = torch.sum(adj,dim=1)
    #     D = torch.pow(D,-0.5)
    #     D = torch.diag(D)
        
    #     filter = torch.matmul(torch.matmul(D,adj),D)
    #     for i in range(k):
    #         x = torch.matmul(filter,x)
    #     return x


    # def forward(self, G_t, G_s, y_t, y_s, A_t, A_s, train, epoch=0, learnA = 0, norm = 0):
        
    #     if learnA:
    #         if train:
    #             A_s = self.discretize(A_s,train, epoch=epoch)
    #         else:
    #             A_s = self.discretize(A_s, train)

    #     if norm:
    #         G_s = F.normalize(G_s, dim=1)
            
    #     G_s = self.GCF(A_s, G_s, self.K)


    #     K_ss      = self.kernel(G_s, G_s, A_s, A_s)
    #     K_ts      = self.kernel(G_t, G_s, A_t, A_s)
    #     n        = torch.tensor(len(G_s), device = G_s.device)
    #     regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device = G_s.device) / n
    #     b         = torch.linalg.solve(K_ss + regulizer, y_s)
    #     pred      = torch.matmul(K_ts, b)

    #     pred      = nn.functional.softmax(pred, dim = 1)
    #     correct   = torch.eq(pred.argmax(1).to(torch.float32), y_t.argmax(1).to(torch.float32)).sum().item()
    #     acc       = correct / len(y_t)
    #     # MSE loss for the sparse matrix A_s
    #     loss_sparse = torch.square(torch.sigmoid(A_s).mean() - 5/70)
    #     return pred, correct, loss_sparse