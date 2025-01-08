import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from torch.nn import functional as F
from krr import KernelRidgeRegression
from sgtk import SimplifyingGraphTangentKernel
from sgnk import SimplifyingGraphNeuralKernel
from sntk import StructureBasedNTK
from ntk import NeuralTangentKernel
from LoadData import load_data
from utils import update_E, sub_E
import argparse
import numpy as np
import random
import time

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='SGNK computation')
parser.add_argument('--dataset', type=str, default="Pubmed", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--ridge', type=float, default=1e-3, help='ridge parameter of KRR (default: 1e-4)')
parser.add_argument('--k', type=int, default=3, help='number of aggr in preprocess the data (default: 1)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 1)')
parser.add_argument('--learn_A', type=int, default=0, help='learning the adjacency matrix (default: 0)')
parser.add_argument('--set_seed', type=bool, default=True, help='setup the random seed (default: True)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--kernel', type=str, default='SGNK', help='kernel method in KRR [SGTK, SGNK] (default: SNTK, the condened set is 0.5*training set)')
args = parser.parse_args()

# K = args.k

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    setup_seed(args.seed)

# load dataset
root = './datasets/'
adj, x, labels, idx_train, _, idx_test,  \
                        x_train, _, x_test, \
                        y_train, _, y_test, \
                        y_train_one_hot, _, y_test_one_hot, _= load_data(root=root, name=args.dataset, k=args.k)

n_class    = len(torch.unique(labels))
n,dim      = x.shape
n_train    = len(y_train)



E_train     = sub_E(idx_train, adj)
E_test      = sub_E(idx_test, adj)

SGTK        = SimplifyingGraphTangentKernel(K = args.k, L = args.L).to(device)
SGNK        = SimplifyingGraphNeuralKernel( L = args.L).to(device)
SNTK        = StructureBasedNTK(K = 2, L = 2).to(device)
ridge       = torch.tensor(args.ridge).to(device)
NTK         = NeuralTangentKernel( L = 2 ).to(device)


def dot_product(x, y, A1, A2):
    return torch.matmul(x,y.T)





if args.kernel == 'SGTK':
    kernel      =  SGTK.nodes_gram
elif args.kernel == "SGNK":
    kernel      =  SGNK.nodes_gram
elif args.kernel == "SNTK":
    kernel      =  SNTK.nodes_gram
elif args.kernel == "dot_product":
    kernel      =  dot_product
elif args.kernel == "NTK":
    kernel      =  NTK.nodes_gram

KRR         = KernelRidgeRegression(kernel,ridge,args.k).to(device)

adj         = adj.to(device)
x           = x.to(device)
x_train     = x_train.to(device)
x_test      = x_test.to(device)
E_test      = E_test.to(device)
E_train     = E_train.to(device)

y_train_one_hot = y_train_one_hot.to(device)
y_test_one_hot = y_test_one_hot.to(device)



def test(G_t, G_s, y_t, y_s, A_t, A_s, Alpha, loss_fn, learnA):
    
    test_loss = 0
    with torch.no_grad():
        pred,acc    = KRR.forward( G_t, G_s, y_t, y_s, A_t, A_s, Alpha, train=False, learnA=learnA)
        test_loss  += loss_fn(pred, y_t).item()

    print(f"Test Acc: {(100*acc):>0.2f}%, Avg loss: {test_loss:>6f}",end = '\n')
    return test_loss, acc*100


Alpha = torch.rand(1).to(device)  # to be optmized parameter of the BinConcrete distribution
Alpha = Alpha * 2


for model in ['gcn', 'sgc', 'appnp', 'sage']:

    if args.dataset ==  'Cora':
        dataset = 'cora'
    elif args.dataset == 'Citeseer':
        dataset = 'citeseer'
    elif args.dataset == 'Pubmed':  
        dataset = 'pubmed'


    print(f"{args.dataset} {model}", end='  ')

    x_s = torch.load(f'GCond_syn_data/{dataset}_X_1.0_{model}.pt').to(device)
    y_s = torch.load(f'GCond_syn_data/{dataset}_Y_1.0_{model}.pt').to(device)
    A_s = torch.load(f'GCond_syn_data/{dataset}_A_1.0_{model}.pt').to(device)


    MSEloss     = nn.MSELoss().to(device)
    y_s = F.one_hot(y_s, n_class).to(torch.float32)
    test_loss, test_acc = test(x_test, x_s, y_test_one_hot, y_s, E_test, A_s, Alpha,  MSEloss, args.learn_A)


