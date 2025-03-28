import torch
from torch import nn
from sklearn.cluster import KMeans
# from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from utils import edge_ind_to_sparse_adj

class FlickrDataLoader(nn.Module):
    def __init__(self, name = 'Flickr', split='train', batch_size=3000, split_method='kmeans', device='cuda:0', k = 2, num_hops=1):
        super(FlickrDataLoader, self).__init__()
        if name == 'Flickr':
            from torch_geometric.datasets import Flickr as DataSet
        elif name == 'Reddit':
            from torch_geometric.datasets import Reddit as DataSet

        Dataset       = DataSet("./datasets/" + name, None, None)
        self.n, self.dim = Dataset[0].x.shape
        self.device    = device
        mask          = split + '_mask'
        features      = Dataset[0].x.to(device)
        labels        = Dataset[0].y
        self.edge_index    = Dataset[0].edge_index

        # values        = torch.ones(edge_index.shape[1])
        # Adj           = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n,self.n]))
        # sparse_eye    = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))
        # self.Adj      = Adj + sparse_eye
        # self.Adj      = self.Adj
        # self.edge_index = self.Adj.coalesce().indices()

        features      = self.normalize_data(features)
        self.split_idx= torch.where(Dataset[0][mask])[0].to(device)
        self.n_split  = len(self.split_idx)
        self.k        = torch.ceil(torch.tensor(self.n_split/batch_size)).to(torch.int)

        # optor_index       = torch.cat((self.split_idx.reshape(1,self.n_split),torch.tensor(range(self.n_split)).reshape(1,self.n_split).to(device)),dim=0)
        # optor_value       = torch.ones(self.n_split).to(device)
        # optor_shape       = torch.Size([self.n,self.n_split])
        # optor             = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        # self.Adj_mask     = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor).to(device)

        _, split_edge_index ,_,_ = k_hop_subgraph(node_idx=self.split_idx, num_hops=0, edge_index=self.edge_index, relabel_nodes=True)
        self.Adj_mask = edge_ind_to_sparse_adj(split_edge_index, self_loop=True).to(device)

        self.split_feat   = features[self.split_idx]

        self.split_feat   = self.GCF(self.Adj_mask, self.split_feat, k = k)

        self.split_label  = labels[self.split_idx].to(device)
        self.split_method = split_method
        self.n_classes    = Dataset.num_classes

        self.batch_labels_list = []
        self.batch_size = batch_size
        # self.num_neighbor = num_neighbor
        self.data         = self.split_feat.cpu()
        self.num_hops     = num_hops

        # data          = Data(x = self.split_feat, edge_index = self.Adj_mask.coalesce().indices(), y = self.split_label)

        # self.train_loader = NeighborLoader(data,
        #                       shuffle=True,
        #                       batch_size=self.batch_size, num_neighbors=self.num_neighbor)






    def loader(self):
        split_edge_index = self.Adj_mask.coalesce().indices()

        # data  = Data(x = self.split_feat, edge_index = split_edge_index, y = self.split_label)

        # data = Data(edge_index=edge_index)
        tensor = torch.arange(self.n_split)
        shuffled_tensor = tensor[torch.randperm(tensor.size(0))]


        groups = torch.chunk(shuffled_tensor, self.k)

        for i, group in enumerate(groups):
            # num_hops = 2  # k-hop

            subset, edge_index_k_hop,_,_ = k_hop_subgraph(node_idx=group, num_hops=self.num_hops, edge_index=split_edge_index, relabel_nodes=True)
            yield Data(x = self.split_feat[subset], edge_index = edge_index_k_hop, y = self.split_label[subset])




    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        mean = data.mean(dim=0)
        std = data.std(dim=0) 
        std[std == 0] = 1 
        normalized_data = (data - mean) / std
        return normalized_data

    def GCF(self, adj, x, k=2):
        """
        Graph convolution filter
        parameters:
            adj: torch.Tensor, adjacency matrix, must be self-looped
            x: torch.Tensor, features
            k: int, number of hops
        return:
            torch.Tensor, filtered features
        """
        n   = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2,1).to(device=adj.device)
        # adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n).to(device=adj.device), (n,n))#.to(device=adj.device)

        D   = torch.pow(torch.sparse.sum(adj,1).to_dense(), -0.5)
        D   = torch.sparse_coo_tensor(ind, D, (n,n))

        matr = torch.sparse.mm(D,adj)
        filter = torch.sparse.mm(matr,D)

        for i in range(k):
            x = torch.sparse.mm(filter,x)
        return x

    def properties(self):
        return self.k, self.n_split, self.n_classes, self.dim, self.n
    
    def split_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'kmeans'
        """

        if self.split_method == 'kmeans':
            kmeans = KMeans(n_clusters = int(self.k))
            kmeans.fit(self.data.numpy())
            self.batch_labels = kmeans.predict(self.data.numpy())
        for i in range(self.k):
            idx       = torch.where(torch.tensor(self.batch_labels) == i)[0]
            self.batch_labels_list.append(idx)

    def getitem(self, idx):
        # idx   = [idx]
        n_idx   = len(idx)
        idx_raw = self.split_idx[idx]
        feat    = self.split_feat[idx]
        label   = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((idx_raw.reshape(1,n_idx),torch.tensor(range(n_idx)).reshape(1,n_idx).to(device=idx_raw.device)),dim=0)
        optor_value = torch.ones(n_idx).to(device=idx_raw.device)
        optor_shape = torch.Size([self.n,n_idx])
        optor       = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape).to(device=idx_raw.device)
        sub_A       = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)

    def get_batch(self, i):
        # idx       = torch.where(torch.tensor(self.batch_labels) == i)[0]
        idx       = self.batch_labels_list[i]
        batch_i   = self.getitem(idx)
        return batch_i
