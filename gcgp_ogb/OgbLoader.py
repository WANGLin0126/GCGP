import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import dataset
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from utils import hierarchical_clustering

class OgbNodeDataLoader(nn.Module):
    def __init__(self, dataset_name='ogbn-arxiv', train_batch_size=500, test_batch_size = 10000, aggr = 2, num_hops = 2, device='cuda:0'):
        super(OgbNodeDataLoader, self).__init__()
        if dataset_name == 'ogbn-mag':
            Dataset             = dataset.NodePropPredDataset(dataset_name, root="./datasets/")
            self.n, self.dim    = Dataset.graph['node_feat_dict']['paper'].shape
            split_set           = Dataset.get_idx_split()
            features            = torch.tensor(Dataset.graph['node_feat_dict']['paper'])
            self.labels         = torch.tensor(Dataset.labels['paper'])
            edge_index     = torch.tensor(Dataset.graph['edge_index_dict'][('paper','cites','paper')])#.to(device)
            self.train_idx      = torch.tensor(split_set['train']['paper']).to(device)
            self.test_idx       = torch.tensor(split_set['test']['paper']).to(device)
            self.device         = device
            norm_features       = self.normalize_data(features)
            values              = torch.ones(edge_index.shape[1])
            Adj                 = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n,self.n]))#.to(device)
            sparse_eye          = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))#.to(device)
            Adj                 = Adj + sparse_eye
            self.Adj            = Adj.to(device)
            self.edge_index     = edge_index.to(device)
            GCF_features        = self.GCF(self.Adj, norm_features, k = aggr).to(device)
            self.features       = GCF_features.to(device)

            
        elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
            Dataset             = dataset.NodePropPredDataset(dataset_name, root="./datasets/")
            self.n, self.dim    = Dataset.graph['node_feat'].shape
            split_set           = Dataset.get_idx_split()
            graph,labels        = Dataset[0]
            features            = torch.tensor(graph['node_feat'])#.to(device)
            edge_index          = torch.tensor(graph['edge_index'])#.to(device)
            values              = torch.ones(edge_index.shape[1])#.to(device)
            Adj                 = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n,self.n]))#.to(device)
            sparse_eye          = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n), (self.n, self.n))#.to(device)
            Adj                 = Adj + sparse_eye
            self.Adj            = Adj.to(device)
            self.edge_index     = edge_index.to(device)
            self.device         = device
            norm_features       = self.normalize_data(features)
            GCF_features        = self.GCF(self.Adj, norm_features, k = aggr)
            self.features       = GCF_features.to(device)
            self.labels         = torch.tensor(labels).to(device)
            self.train_idx      = torch.tensor(split_set['train']).to(device)
            self.test_idx       = torch.tensor(split_set['test']).to(device)
        else:
            raise ValueError('Dataset not supported')
        
        self.n_train            = len(self.train_idx)
        self.n_test             = len(self.test_idx)
        self.train_batch_size   = train_batch_size
        self.test_batch_size    = test_batch_size
        self.k                  = torch.ceil(torch.tensor(self.n_test/test_batch_size)).to(torch.int)
        self.n_classes          = Dataset.num_classes
        # self.num_neighbor       = num_neighbor
        self.batch_labels_list  = []
        self.train_feat         = self.features[self.train_idx]
        self.test_feat          = self.features[self.test_idx]
        self.test_label         = self.labels[self.test_idx]
        self.train_label        = self.labels[self.train_idx]
        self.train_Adj          = self.sub_A(self.Adj,self.train_idx)
        self.test_Adj           = self.sub_A(self.Adj,self.test_idx)
        self.train_batch        = torch.ceil(torch.tensor(self.n_train/train_batch_size)).to(torch.int)
        self.test_batch         = torch.ceil(torch.tensor(self.n_test/test_batch_size)).to(torch.int)
        self.num_hops           = num_hops
        

    def train_loader(self):
        train_edge_index = self.train_Adj.coalesce().indices()

        data  = Data(x = self.train_feat, edge_index = train_edge_index, y = self.train_label)

        # data = Data(edge_index=edge_index)
        tensor = torch.arange(self.n_train)
        shuffled_tensor = tensor[torch.randperm(tensor.size(0))]

        # 根据 training batch size 分成 self.train_batch 组
        groups = torch.chunk(shuffled_tensor, self.train_batch)

        for i, group in enumerate(groups):
            # num_hops = 2  # k-hop
            # 使用 k_hop_subgraph 函数查找 k-hop 邻居， 输出的是子图的节点索引 和 边索引
            subset, edge_index_k_hop,_,_ = k_hop_subgraph(node_idx=group, num_hops=self.num_hops, edge_index=data.edge_index, relabel_nodes=True)
            # 生成 training batch 数据
            yield Data(x = self.train_feat[subset], edge_index = edge_index_k_hop, y = self.train_label[subset])


    def test_loader(self):
        test_edge_index = self.test_Adj.coalesce().indices()

        data  = Data(x = self.test_feat, edge_index = test_edge_index, y = self.test_label)

        # data = Data(edge_index=edge_index)
        tensor = torch.arange(self.n_test)
        shuffled_tensor = tensor[torch.randperm(tensor.size(0))]

        # 根据 test batch size 分成 self.test_batch 组
        groups = torch.chunk(shuffled_tensor, self.test_batch)

        for i, group in enumerate(groups):
            # num_hops = 2  # k-hop
            # 使用 k_hop_subgraph 函数查找 k-hop 邻居， 输出的是子图的节点索引 和 边索引
            subset, edge_index_k_hop,_,_ = k_hop_subgraph(node_idx=group, num_hops=0, edge_index=data.edge_index, relabel_nodes=True)
            # 生成 training batch 数据
            yield Data(x = self.test_feat[subset], edge_index = edge_index_k_hop, y = self.test_label[subset])
    # def get_train_batch(self, idx):

    #     n_idx   = len(idx)
    #     # Adj     = self.Adj.to(self.device)
    #     # idx_raw = self.train_idx[idx].to(self.device)
    #     feat    = self.train_feat[idx].to(self.device)
    #     label   = self.train_label[idx].to(self.device)
    #     # idx   = idx.tolist()

    #     optor_index = torch.cat((idx.reshape(1,n_idx),torch.tensor(range(n_idx)).reshape(1,n_idx).to(self.device)),dim=0)
    #     optor_value = torch.ones(n_idx).to(self.device)
    #     # optor_shape = torch.Size([self.n,n_idx]).to(self.device)

    #     optor       = torch.sparse_coo_tensor(optor_index, optor_value, [self.n,n_idx]).to(self.device)
    #     sub_A       = torch.sparse.mm(torch.sparse.mm(optor.t(), self.train_Adj), optor)

    #     return (feat, label, sub_A)



    def sub_A(self, Adj, idx):
        '''
        gien the index of the nodes [idx], and the adjacency matrix [Adj] return the sub-adjacency matrix
        '''

        # idx         = torch.tensor(idx)
        n_idx       = len(idx)
        optor_index = torch.cat((idx.reshape(1,n_idx),torch.tensor(range(n_idx)).reshape(1,n_idx).to(self.device)),dim=0)
        optor_value = torch.ones(n_idx).to(self.device)
        optor       = torch.sparse_coo_tensor(optor_index, optor_value, [self.n,n_idx])
        sub_A       = torch.sparse.mm(torch.sparse.mm(optor.t(), Adj), optor)
        return sub_A




    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        data = data.to(self.device)

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
        adj = adj.to(self.device)
        x = x.to(self.device)
        n = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2,1).to(adj.device)
        # adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n).to(adj.device), (n,n))
        D = torch.pow(torch.sparse.sum(adj,1).to_dense(), -0.5)
        D = torch.sparse_coo_tensor(ind, D, (n,n))
        filter = torch.sparse.mm(torch.sparse.mm(D,adj),D)


        for i in range(k):
            x = torch.sparse.mm(filter,x)
        return x

    def properties(self):
        return self.train_batch, self.test_batch, self.n_train, self.n_test, self.n_classes, self.dim, self.n
    
    def split_test_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'random'
        """
        # data = self.test_feat.to(self.device)
        # data = self.split_feat.cpu()
        # if self.split_method == 'kmeans':
        #     # kmeans = KMeans(n_clusters = self.k)
        #     # kmeans.fit(data.numpy())
        #     # self.batch_labels = kmeans.predict(data.numpy())

        #     # self.batch_labels,_ = kmeans_large_scale(data, self.k, batch_size=20000)
        #     # self.batch_labels,_ = kmeans_balanced(data, self.k, batch_size=20000)
        #     self.batch_labels = hierarchical_clustering(data, self.clustering_layers)

        #     for i in range(self.k):
        #         idx       = torch.where(torch.tensor(self.batch_labels) == i)[0]
        #         self.batch_labels_list.append(idx)
        # else:
        for i in range(self.k):
            if i == self.k - 1:
                idx = torch.tensor(range(i*self.test_batch_size, self.n_test))
            else:
                idx = torch.tensor(range(i*self.test_batch_size, (i+1)*self.test_batch_size))
            self.batch_labels_list.append(idx)
        
        # self.batch_labels_list = self.batch_labels_list.cpu()
        # save batch labels
        # torch.save(self.batch_labels, './{}_{}_batch_labels.pt'.format(self.split_method, self.k))

    def getitem(self, idx):
        # idx   = [idx]
        n_idx   = len(idx)
        Adj     = self.Adj.to(self.device)
        idx_raw = self.test_idx[idx].to(self.device)
        feat    = self.test_feat[idx].to(self.device)
        label   = self.test_label[idx].to(self.device)
        # idx   = idx.tolist()

        optor_index = torch.cat((idx_raw.reshape(1,n_idx),torch.tensor(range(n_idx)).reshape(1,n_idx).to(self.device)),dim=0)
        optor_value = torch.ones(n_idx).to(self.device)
        # optor_shape = torch.Size([self.n,n_idx]).to(self.device)

        optor       = torch.sparse_coo_tensor(optor_index, optor_value, [self.n,n_idx]).to(self.device)
        sub_A       = torch.sparse.mm(torch.sparse.mm(optor.t(), Adj), optor)

        return (feat, label, sub_A)

    def get_test_batch(self, i):
        # idx       = torch.where(torch.tensor(self.batch_labels) == i)[0]
        idx       = self.batch_labels_list[i]
        batch_i   = self.getitem(idx)
        return batch_i

    # def kmeans_centers(self):
    #     """
    #     genrate kmeans centers
    #     """
    #     data = self.split_feat.cpu()
    #     kmeans = KMeans(n_clusters = self.M)
    #     kmeans.fit(data.numpy())
    #     centers = kmeans.cluster_centers_
    #     return centers