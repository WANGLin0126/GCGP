"""
Training different GNNs models 
Datasets: Cora, Pubmed, Computers, Photo
GNNs models: GAT, SAGE, SGC, MLP, Cheby, GCN, APPNPModel, GIN


python gnn_nodes_classification.py  --dataset Cora --model GAT --hidden_size 128 --num_heads 8 --epochs 100 --lr 1e-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import GAT, GraphSAGE, SGC, MLP, Cheby, GCN, APPNPModel, GIN
import argparse
import numpy as np
import random
from torch_geometric.datasets import Planetoid, Amazon
import scipy.sparse as sp
import time
from tqdm import tqdm



parser = argparse.ArgumentParser(description='GNN training')
parser.add_argument('--model', type=str, default="GIN", help='name of model [GAT, SAGE, SGC, MLP, Cheby, GCN, APPNP] (default: GCN)')
parser.add_argument('--dataset', type=str, default="Computers", help='name of dataset [Cora, Citeseer, Pubmed, Computers, Photo](default: Cora)')
parser.add_argument('--alpha', type=float, default=0.2, help=' the parameter of APPNP (default: 0.2)')
parser.add_argument('--num_layers', type=int, default=2, help=' the parameter of APPNP (default: 2)')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate in GAT and APPNP (default: 0.05)')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--set_seed', type=bool, default=True)
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--iter', type=int, default=3)
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    print('Set seed')
    setup_seed(args.seed)




def normalize_data(data):
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


def GCF(adj, x, k=1):
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


def load_data(root,name):
    if name in ['Cora','Citeseer','Pubmed','CS','Physics']:
        dataset    = Planetoid(root=root,name=name,split='public')
        # # dataset    = CitationFull(root=root,name=name)
        # # dataset    = Coauthor(root=root,name=name)
        # dataset    = AmazonProducts(root=root)
        train_mask = dataset[0]['train_mask']
        val_mask   = dataset[0]['val_mask']
        test_mask  = dataset[0]['test_mask']
        x          = dataset[0]['x']           # all features
        y          = dataset[0]['y']           # all labels
    elif name in ['Computers','Photo']:
        dataset    = Amazon(root=root,name=name)
        x          = dataset[0]['x']           # all features
        y          = dataset[0]['y']           # all labels
        n_class    = len(torch.unique(y))
        n,_        = x.shape
        idx_train = []
        idx_test  = []
        for i in range(n_class):
            idx = torch.where(y==i)[0]
            idx_train.append(idx[:20])
            idx_test.append(idx[20:120])
        idx_train = torch.cat(idx_train)
        idx_test  = torch.cat(idx_test)
        train_mask = torch.zeros(n,dtype=torch.bool)
        test_mask  = torch.zeros(n,dtype=torch.bool)
        train_mask[idx_train] = True
        test_mask[idx_test]   = True
        val_mask = ~ (train_mask | test_mask)
    else:
        raise ValueError('Dataset not found!')



    edge_index = dataset[0]['edge_index']
    n_class    = len(torch.unique(y))
    n,_      = x.shape

    adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), edge_index), shape=(n, n)).toarray()
    adj = torch.tensor(adj)
    adj = adj + torch.eye(adj.shape[0])  

    # x = normalize_data(x)
    # x = GCF(adj, x, k=2) 


    x_train    = x[train_mask]
    x_val      = x[val_mask]
    x_test     = x[test_mask]

    y_train    = y[train_mask]
    y_val      = y[val_mask]
    y_test     = y[test_mask]

    idx_train = torch.where(train_mask)[0]
    idx_val   = torch.where(val_mask)[0]
    idx_test  = torch.where(test_mask)[0]

    y_one_hot       = F.one_hot(y, n_class)
    y_train_one_hot = y_one_hot[train_mask]
    y_val_one_hot   = y_one_hot[val_mask]
    y_test_one_hot  = y_one_hot[test_mask]

    return edge_index, x, y, y_one_hot, train_mask, test_mask



root = './datasets/'
edge_index, x, _, y, train_mask, test_mask = load_data(root=root, name=args.dataset)

edge_index = edge_index.to(device)
x = x.to(device)
y = y.to(device).to(torch.float32)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
num_classes    = y.shape[1]
n,num_features      = x.shape






# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

results = torch.zeros(args.epochs, args.iter).to(device)
Time = torch.zeros(args.epochs, args.iter).to(device)

max_test_acc = 0

print('Model', args.model)


for iter in range(args.iter):
    
    if args.model == 'GAT':
        model = GAT(num_features, args.hidden_size, num_classes, args.num_heads, args.dropout)
    elif args.model == 'SAGE':
        model = GraphSAGE(num_features, args.hidden_size, num_classes)
    elif args.model == 'SGC':
        model = SGC(num_features, num_classes)
    elif args.model == 'MLP':
        model = MLP(num_features, args.hidden_size, num_classes)
    elif args.model == 'Cheby':
        model = Cheby(num_features, args.hidden_size, num_classes, 2)
    elif args.model == 'GCN':
        model = GCN(num_features, args.hidden_size, num_classes)
    elif args.model == 'APPNP':
        model = APPNPModel(num_features, args.hidden_size, num_classes, args.num_layers, args.alpha, args.dropout)
    elif args.model == 'GIN':
        model = GIN(num_features, args.hidden_size, num_classes)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    print('--------------------------------------------------')
    print('The '+str(iter+1)+'th iteration:')
    print('--------------------------------------------------')

    a = time.time()
    Max_Acc = 0
    for epoch in range(args.epochs):

        # training
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        output = model(x, edge_index)
        loss = criterion(output[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        total_loss = loss.item()
        print('Epoch {}, Training Loss: {:.4f}'.format(epoch + 1, total_loss), end=' ')

        # testing
        model.eval()
        pred = model(x, edge_index)
        correct = pred[test_mask].argmax(dim=1).eq(y[test_mask].argmax(dim=1)).sum().item()
        loss = criterion(pred[test_mask], y[test_mask].to(torch.float32))

        test_loss = loss.item()
        accuracy = correct / test_mask.sum().item()

        b = time.time()
        results[epoch][iter] = accuracy
        Time[epoch][iter] = b - a
        Max_Acc = max(Max_Acc, accuracy)

        print('Test Loss: {:.4f}'.format(test_loss),'Test Acc: {:.4f}'.format(accuracy), end='\n')
    max_test_acc = max(Max_Acc, max_test_acc)




# mean and std
results_mean = torch.mean(results, dim=1)
Time_mean = torch.mean(Time, dim=1)
results_std  = torch.std(results, dim=1)



max_mean, max_mean_index = torch.max(results_mean, dim=0)



print(f"Dataset       :{args.dataset}")
print(f"Classes       :{num_classes}")
print(f"Dim           :{num_features}")
print(f"Model         :{args.model}")
print(f"Layers        :{args.num_layers}")
print(f"Wide          :{args.hidden_size}")
print(f"LR            :{args.lr}")


print('The max mean of test accuracy: {:.4f} at epoch {}'.format(max_mean, max_mean_index+1))
print('The std of test accuracy     : {:.4f}'.format(results_std[max_mean_index]))
print('Training time used           : {:.4f}'.format(Time_mean[max_mean_index]))
print('Max Test Accuracy            : {:.4f}'.format(max_test_acc))
# print(results[max_mean_index])


