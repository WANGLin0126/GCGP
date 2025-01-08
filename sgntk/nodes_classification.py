"""
nodes classification test by using the KRR model and the proposed sgntk and sntk kernels

With the normalization of the data
python nodes_classification.py --dataset Cora --kernel SNTK --ridge 0.1 --k 4
python nodes_classification.py --dataset Cora --kernel SGNTK --ridge 0.6 --k 3

python nodes_classification.py --dataset Pubmed --kernel SGNTK --ridge 0.9 --k 3
python nodes_classification.py --dataset Pubmed --kernel SNTK --ridge 0.9 --k 3

python nodes_classification.py --dataset Photo --kernel SNTK --ridge 0.2 --k 2
python nodes_classification.py --dataset Photo --kernel SGNTK --ridge 0.2 --k 2

python nodes_classification.py --dataset Computers --kernel SGNTK --ridge 0.2 --k 2
python nodes_classification.py --dataset Computers --kernel SNTK --ridge 0.01 --k 2

python nodes_classification.py --dataset Citeseer --kernel SNTK --ridge 1 --k 3
python nodes_classification.py --dataset Citeseer --kernel SGNTK --ridge 1 --k 3


Without the normalization of the data
python nodes_classification.py --dataset Computers --kernel SGNTK --ridge 0.01 --k 2
python nodes_classification.py --dataset Computers --kernel SNTK --ridge 0.01 --k 2

python nodes_classification.py --dataset Photo --kernel SNTK --ridge 0.05 --k 2
python nodes_classification.py --dataset Photo --kernel SGNTK --ridge 0.01 --k 2

python nodes_classification.py --dataset Pubmed --kernel SGNTK --ridge 1 --k 3
python nodes_classification.py --dataset Pubmed --kernel SNTK --ridge 1 --k 3

python nodes_classification.py --dataset Cora --kernel SNTK --ridge 0.5 --k 3
python nodes_classification.py --dataset Cora --kernel SGNTK --ridge 0.8 --k 3

python nodes_classification.py --dataset Citeseer --kernel SNTK --ridge 1 --k 3
python nodes_classification.py --dataset Citeseer --kernel SGNTK --ridge 1 --k 3
"""



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from torch.nn import functional as F
from sgntk_planetoid.sgtk import StructureBasedNeuralTangentKernel
from sgntk_planetoid.sgnk import SimplifyingGraphNeuralTangentKernel
from sntk import StructureBasedNTK
from LoadData import load_data
from utils import update_E, sub_E
import argparse
import numpy as np
import random
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='SNTK computation')
parser.add_argument('--dataset', type=str, default="Citeseer", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--ridge', type=float, default=1e0, help='ridge parameter of KRR (default: 1e-4)')
parser.add_argument('--k', type=int, default=3, help='number of aggr in kernel method (default: 1)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 1)')
parser.add_argument('--kernel', type=str, default='RBF', help='kernel method in KRR [SNTK, SGNTK, NTK, RBF, StruNTK] (default: SNTK)')
args = parser.parse_args()


def KRR(G_t, G_s, y_t, y_s, A_t, A_s, kernel, ridge):
    K_ss      = kernel(G_s, G_s, A_s, A_s)
    K_ts      = kernel(G_t, G_s, A_t, A_s)
    n        = torch.tensor(len(G_s), device = G_s.device)
    regulizer = ridge * torch.trace(K_ss) * torch.eye(n, device = G_s.device) / n
    b         = torch.linalg.solve(K_ss + regulizer, y_s)
    pred      = torch.matmul(K_ts, b)
    # pred      = torch.sigmoid(pred)
    # pred      = nn.functional.softmax(pred, dim = 1)
    correct   = torch.eq(pred.argmax(1).to(torch.float32), y_t.argmax(1).to(torch.float32)).sum().item()
    acc       = correct / len(y_t)

    return pred, acc



def RBF(G_t, G_s, A_t, A_s):
    G_t = G_t.unsqueeze(1)
    G_s = G_s.unsqueeze(0)
    return torch.exp(-torch.pow((G_t - G_s)/10, 2).sum(2))



def RBF2(X, Y,  A_t, A_s, sigma=10):
    """
    Compute the RBF (Gaussian) kernel matrix between two sets of samples.
    
    Parameters:
    X (torch.Tensor): The first set of samples, of shape (N, D).
    Y (torch.Tensor): The second set of samples, of shape (M, D).
    sigma (float): The bandwidth parameter of the RBF kernel.
    
    Returns:
    torch.Tensor: The RBF kernel matrix of shape (N, M).
    """
    N, D = X.shape
    M, _ = Y.shape
    
    # Compute the squared Euclidean distance matrix
    dist_matrix = torch.sum((X[:, None, :] - Y[None, :, :]) ** 2, dim=2)
    
    # Apply the RBF kernel function
    kernel_matrix = torch.exp(-dist_matrix / (2 * sigma ** 2))
    
    return kernel_matrix


if args.kernel == 'SNTK':
    SNTK        = StructureBasedNeuralTangentKernel( L=args.L).to(device)
    kernel = SNTK.nodes_gram
elif args.kernel == 'SGNTK':    
    SGNTK       = SimplifyingGraphNeuralTangentKernel( L=args.L).to(device)
    kernel = SGNTK.nodes_gram
elif    args.kernel == 'RBF':
    kernel = RBF
    args.k = 0
elif args.kernel == 'NTK':
    NTK        = StructureBasedNeuralTangentKernel( L=args.L).to(device)
    kernel = NTK.nodes_gram
    args.k = 0
elif args.kernel == 'StruNTK':
    StruNTK        = StructureBasedNTK(K=2, L=2).to(device)
    kernel = StruNTK.nodes_gram
elif args.kernel == 'RBF2':
    kernel = RBF2
    args.k = 0
else:
    raise ValueError('Kernel not found!')




# load dataset
root = './datasets/'
adj, x, labels, idx_train, _, idx_test,  \
                        x_train, _, x_test, \
                        y_train, _, y_test, \
                        y_train_one_hot, _, y_test_one_hot, _= load_data(root=root, name=args.dataset, k=args.k)

n_class    = len(torch.unique(labels))
n,dim      = x.shape


print(f"Dataset       :{args.dataset}")
print(f"Training Set  :{len(y_train)}")
print(f"Testing Set   :{len(y_test)}")
print(f"Classes       :{n_class}")
print(f"Dim           :{dim}")
print(f"Ridge         :{args.ridge}")
print(f"k             :{args.k}")
print(f"L             :{args.L}")
print(f"Kernel        :{args.kernel}")



x_test = x_test.to(device)
x_train = x_train.to(device)
y_test = y_test.to(device)
y_train = y_train.to(device)
y_train_one_hot = y_train_one_hot.to(torch.float32).to(device)
y_test_one_hot = y_test_one_hot.to(torch.float32).to(device)
A_train = sub_E(idx_train, adj).to(device)
A_test  = sub_E(idx_test, adj).to(device)



# kernel      = SNTK.nodes_gram if args.kernel == 'SNTK' else SGNTK.nodes_gram
ridge       = torch.tensor(args.ridge).to(device)
Time  = []
for i in range(10):
    time1 = time.time()
    pred,acc = KRR(x_test, x_train, y_test_one_hot, y_train_one_hot, A_test, A_train, kernel, ridge)
    time2 = time.time()
    Time.append(time2-time1)
print(f"Accuracy      :{acc}")
print(f"Time          :{np.mean(Time)}")