"""
training different gnn models on condensed data obetained by FastGC method
the training data is just loaded from the saved files in /save/ folder
the test data is the original data with graph convolutional filter applied

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from models import GAT, GraphSAGE, SGC, MLP, Cheby, GCN, APPNPModel
import argparse
import numpy as np
import random
# from LoadData import load_data

# python main.py  --alpha 0.2 --num_layers 2 --dropout 0.05 --hidden_size 128 --num_heads 8 --epochs 200 --lr 1e-3  --dataset Cora --kernel SGNTK --k 4 --model GCN

parser = argparse.ArgumentParser(description='FastGC Generalization')
parser.add_argument('--model', type=str, default="SGC", help='name of model [GAT, SAGE, SGC, MLP, Cheby, GCN, APPNPModel] (default: GCN)')
parser.add_argument('--dataset', type=str, default="Citeseer", help='name of dataset [Cora, Citeseer, Pubmed, Computers, Photo, CS, Physics](default: Cora)')
parser.add_argument('--cond_ratio', type=float, default=1, help='condensation ratio (default: 0.25)')
parser.add_argument('--learnA', type=int, default=0, help='learn adjacency matrix or not (default: 0)')
parser.add_argument('--alpha', type=float, default=0.2, help=' the parameter of APPNP (default: 0.2)')
parser.add_argument('--num_layers', type=int, default=2, help=' the parameter of APPNP (default: 2)')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate in GAT and APPNP (default: 0.05)')
parser.add_argument('--kernel', type=str, default='SNTK', help='kenrel method')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--set_seed', type=bool, default=True)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--iter', type=int, default=5)
parser.add_argument('--k', type=int, default=3)
args = parser.parse_args()

# if args.dataset == 'Cora':
#     cond_size=70
# elif args.dataset == 'Pubmed':
#     cond_size=30
# elif args.dataset == 'Photo':
#     cond_size=80
# elif args.dataset == 'Computers':
#     cond_size=100
# elif args.dataset == 'ogbn-arxiv':
#     cond_size=90
# elif args.dataset == 'Reddit':
#     cond_size=77



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    print('Set seed')
    setup_seed(args.seed)






from torch_geometric.datasets import Planetoid, Amazon, CitationFull,Coauthor, Yelp, AmazonProducts
from torch.nn import functional as F
import scipy.sparse as sp


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


def load_data(root,name,k):
    if name in ['Cora','Citeseer','Pubmed']:
        dataset    = Planetoid(root=root,name=name,split='public')
        # # dataset    = CitationFull(root=root,name=name)
        train_mask = dataset[0]['train_mask']
        val_mask   = dataset[0]['val_mask']
        test_mask  = dataset[0]['test_mask']
        x          = dataset[0]['x']           # all features
        y          = dataset[0]['y']           # all labels
    elif name in ['CS','Physics']:
        dataset    = Coauthor(root=root,name=name)
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
    x = GCF(adj, x, k) 

    idx_train = torch.where(train_mask)[0]
    idx_test  = torch.where(test_mask)[0]

    return edge_index, x, y, test_mask






# load dataset
root = './datasets/'
edge_index, x, y, test_mask = load_data(root=root, name=args.dataset, k=args.k)
n_test          = test_mask.sum().item()
num_classes     = len(torch.unique(y))
num_features    = x.shape[1]


edge_index = edge_index.to(device)
x = x.to(device)
y = y.to(device)
test_mask = test_mask.to(device)



criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()




x_s = torch.load('./save/'+args.dataset+'_x_s_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learnA)+'.pt', map_location=torch.device('cpu')).to(device)
y_s = torch.load('./save/'+args.dataset+'_y_s_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learnA)+'.pt', map_location=torch.device('cpu')).to(device)
A_s = torch.load('./save/'+args.dataset+'_A_s_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learnA)+'.pt', map_location=torch.device('cpu')).to(device)
# pred_x_s = torch.load('./data/'+args.dataset+'_'+args.kernel+'_pred_'+str(cond_size)+'.pt', map_location=torch.device('cpu')).to(device)
# edge_index = A_s.coalesce().indices().to(device)

# size = A_s.shape[0]
# A_s = torch.sigmoid(A_s)
# threshold = 0.5
# A_s[A_s  > threshold] = 1
# A_s[A_s <= threshold] = 0

ind = torch.where(A_s==1)
edge_index_A_s = torch.cat((ind[0].unsqueeze(0),ind[1].unsqueeze(0)), dim=0)
# edge_index_A_s = A_s


y_s_one_hot = F.one_hot((y_s).argmax(dim=1), num_classes).to(torch.float32)


results = torch.zeros(args.epochs, args.iter).to(device)


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

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    print('--------------------------------------------------')
    print('The '+str(iter+1)+'th iteration:')
    print('--------------------------------------------------')


    
    for epoch in range(args.epochs):

        # training
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        output = model(x_s, edge_index_A_s)
        # loss = criterion(output, y_s_one_hot)
        loss = criterion(output, y_s)
        # loss = criterion(output, pred_x_s)


        loss.backward()
        optimizer.step()
        total_loss = loss.item()
        # print('Epoch {}, Training Loss: {:.4f}'.format(epoch + 1, total_loss), end=' ')

        # testing
        model.eval()
        correct = 0


        pred = model(x, edge_index).argmax(dim=1)
        correct += pred[test_mask].eq(y[test_mask]).sum().item()
        pred_one_hot = F.one_hot(pred, num_classes).to(torch.float32)
        y_one_hot = F.one_hot(y, num_classes).to(torch.float32)
        loss = criterion(pred_one_hot[test_mask], y_one_hot[test_mask].to(torch.float32))
        

        test_loss = loss.item()
        accuracy = correct / n_test
        results[epoch][iter] = accuracy
        max_test_acc = max(max_test_acc, accuracy)

        # print('Test Loss: {:.4f}'.format(test_loss),'Test Acc: {:.4f}'.format(accuracy), end='\n')

    
# mean and std
results_mean = torch.mean(results, dim=1).cpu()
results_std  = torch.std(results, dim=1).cpu()

max_mean, max_mean_index = torch.max(results_mean, dim=0)
epochs = torch.tensor(range(args.epochs)) + 1
epoch_Acc = torch.cat((epochs.unsqueeze(1), results_mean.unsqueeze(1), results_std.unsqueeze(1)), dim=1)

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

print(np.array(epoch_Acc.cpu()))

print(f"Dataset       :{args.dataset}")
print(f"Model         :{args.model}")
print(f"Conden ratio  :{args.cond_ratio}")
print(f"GC            :{args.k}")
print(f"LearnA        :{args.learnA}")

print('Max Test Accuracy:')
print(max_test_acc)
print(f'Best Result: Epoch   Acc  Std.')   
print(f'               {max_mean_index+1} {max_mean.item():>0.4f}  {results_std[max_mean_index]:>0.4f}')


