import torch
from torch_geometric.nn import MessagePassing
import numpy as np
"""
Transform a single, large graph with n nodes to n subgraphs
"""

def find(idx,A,c=0):
    """
    find out the one-hop neighbors of nodes in given idx
    len(list) = len(idx)
    A must tensor
    """
    list = []
    for node in idx:
        neigh = torch.where(A[node]==1)[0]
        if c :
            list.append(neigh)
        else:
            for i in range(len(neigh)):
                list.append(neigh[i])
    if c:
        return list
    else:
        return torch.unique(torch.tensor(list))


def find_hop_idx(i,j,A):
    """
    find the index of the j-hop neighbors of node i
    """
    idx = [i,]
    for hop in range(j):
        idx = find(idx,A)
    return idx


def sub_G(A, hop):
    """
    A is the adjacency martrix of graph
    """
    n         = A.shape[0]
    neighbors = []
    for i in range(n):
        neighbor_i = torch.tensor(find_hop_idx(i,hop,A))
        neighbors.append(neighbor_i.to(A.device))
    return neighbors



def sub_A_list(neighbors, A):
    """
    output the adjacency matrix of subgraph
    """
    n        = A.shape[0]
    sub_A_list= []
    for node in range(n):

        n_neig   = len(neighbors[node])
        operator = torch.zeros([n,n_neig]).to(A.device)
        operator[neighbors[node],range(n_neig)] = 1
        sub_A = torch.matmul(torch.matmul(operator.t(),A),operator)
        sub_A_list.append(sub_A)

    return sub_A_list


def sub_A(idx, A):
    """
    output the adjacency matrix of subgraph of idx
    """
    n        = A.shape[0]
    n_neig   = len(idx)
    operator = torch.zeros([n,n_neig])
    operator[idx,range(n_neig)] = 1
    sub_A    = torch.matmul(torch.matmul(operator.t(),A),operator)

    return sub_A


def sub_E(idx, A):
    """
    output the adjacency matrix of subgraph of idx
    """
    n        = A.shape[0]
    n_neig   = len(idx)
    operator = torch.zeros([n,n_neig])
    operator[idx,range(n_neig)] = 1
    sub_A    = torch.matmul(torch.matmul(operator.t(),A),operator)

    ind    = torch.where(sub_A!=0)
    inds   = torch.cat([ind[0],ind[1]]).reshape(2,len(ind[0]))
    values = torch.ones(len(ind[0]))
    sub_E  = torch.sparse_coo_tensor(inds, values, torch.Size([n_neig, n_neig])).to(A.device)

    return sub_E



def update_A(x_s,neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n,n)
    A = torch.zeros(n*n)

    for i in range(n):
        for j in range(i,n):
            K[i,j] = torch.norm(x_s[i]-x_s[j])
            K[j,i] = K[i,j]
    
    edge  = int(n+torch.round(torch.tensor(neig*n/2)))          # the edge number, must be even
    
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)               
    _, indices = torch.sort(Simil) # sort all the similarities
    A[indices[0:edge]] = 1              
    A = A.reshape(n,n).to(x_s.device)

    return A



def update_E(x_s,neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n,n)
    A = torch.zeros(n*n)

    for i in range(n):
        for j in range(i,n):
            K[i,j] = torch.norm(x_s[i]-x_s[j])
            K[j,i] = K[i,j]
    
    edge  = int(n+torch.round(torch.tensor(neig*n/2)))          # the edge number, must be even
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)               
    _, indices = torch.sort(Simil) # sort all the similarities
    A[indices[0:edge]] = 1              
    A = A.reshape(n,n)
    ind = torch.where(A==1)

    ind = torch.cat([ind[0],ind[1]]).reshape(2,edge)
    values = torch.ones(edge)
    E = torch.sparse_coo_tensor(ind, values, torch.Size([n,n])).to(x_s.device)

    return E
    
class Aggr(MessagePassing):
    """
    Undirected nodes features aggregation ['add', 'mean']
    """
    def __init__(self, aggr='add'):
        super(Aggr, self).__init__(aggr=aggr)

    def forward(self, x, edge_index):
        """
        inputs:
            x: [N, dim]
            edge_index: [2, edge_num]
        outputs:
            the aggregated node features
            out: [N, dim]
        """
        edge_index = torch.cat([edge_index, edge_index.flip(dims=[0])], dim = 1)
        edge_index = torch.unique(edge_index, dim = 1)
        return self.propagate(edge_index, x=x) + x
    




def edge_ind_to_sparse_adj(edge_index, self_loop=True):
    """
    convert edge_index to self-looped sparse adjacency matrix
    """
    n = torch.max(edge_index)+1
    values        = torch.ones(edge_index.shape[1]).to(edge_index.device)
    Adj           = torch.sparse_coo_tensor(edge_index, values, torch.Size([n,n]))

    if self_loop:
        sparse_eye = torch.sparse_coo_tensor(torch.arange(n).repeat(2, 1), torch.ones(n), (n, n)).to(edge_index.device)
        Adj = Adj + sparse_eye
    return Adj



import torch.nn.functional as F
from collections import deque, Counter

def kmeans(data, num_clusters=2, max_iters=20, tol=1e-4):
    N, D = data.size()
    indices = torch.randperm(N)[:num_clusters]
    centroids = data[indices].clone()
    for i in range(max_iters):
        centroids_old = centroids.clone()

        distances = torch.cdist(data, centroids)  # (N, num_clusters)
        labels = torch.argmin(distances, dim=1)

        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                centroids[k] = data[mask].mean(dim=0)
            else:

                centroids[k] = data[torch.randint(0, N, (1,))]

        diff = torch.norm(centroids - centroids_old, dim=1).sum()
        if diff < tol:
            break
    return labels

def hierarchical_clustering(data, num_layers, max_iters=20, tol=1e-4):

    N = data.size(0)
    device = data.device

    labels = torch.full((N,), -1, dtype=torch.long, device=device)

    cluster_queue = deque()

    next_cluster_id = 0

    cluster_queue.append((torch.arange(N, device=device), 0, next_cluster_id))
    next_cluster_id += 1

    while cluster_queue:
        indices, depth, cluster_id = cluster_queue.popleft()
        if depth >= num_layers:

            labels[indices] = cluster_id
            continue
        cluster_data = data[indices]

        sub_labels = kmeans(cluster_data, num_clusters=2, max_iters=max_iters, tol=tol)

        for k in range(2):
            mask = (sub_labels == k)
            sub_indices = indices[mask]
            if sub_indices.size(0) == 0:
                continue

            new_cluster_id = next_cluster_id
            next_cluster_id += 1
            labels[sub_indices] = new_cluster_id

            cluster_queue.append((sub_indices, depth + 1, new_cluster_id))
    return labels