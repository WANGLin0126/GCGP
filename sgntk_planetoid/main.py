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
parser.add_argument('--dataset', type=str, default="Photo", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--cond_ratio', type=float, default=0.5, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
parser.add_argument('--ridge', type=float, default=1e-3, help='ridge parameter of KRR (default: 1e-4)')
# parser.add_argument('--K', type=int, default=3, help='number of aggr in kernel method (default: 1)')
parser.add_argument('--k', type=int, default=0, help='number of aggr in preprocess the data (default: 1)')
parser.add_argument('--L', type=int, default=2, help='the number of layers after each aggr (default: 1)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 100)')
parser.add_argument('--lr_X', type=float, default=0.01, help='learning rate (default: 0.005)')
parser.add_argument('--learn_A', type=int, default=0, help='learning the adjacency matrix (default: 0)')
parser.add_argument('--set_seed', type=bool, default=True, help='setup the random seed (default: True)')
parser.add_argument('--save', type=int, default=0, help='save the results (default: False)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--iter', type=int, default=5, help='iteration times of the experiments (default: 5)')
parser.add_argument('--kernel', type=str, default='dot_product', help='kernel method in KRR [SGTK, SGNK] (default: SNTK, the condened set is 0.5*training set)')
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
Cond_size  = round(n_train*args.cond_ratio)
idx_s      = torch.tensor(range(Cond_size))

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
MSEloss     = nn.MSELoss().to(device)

adj         = adj.to(device)
x           = x.to(device)
x_train     = x_train.to(device)
x_test      = x_test.to(device)
E_test      = E_test.to(device)
E_train     = E_train.to(device)

y_train_one_hot = y_train_one_hot.to(device)
y_test_one_hot = y_test_one_hot.to(device)

lr_A = args.lr_X


print(f"Dataset     :{args.dataset}")
print(f"Epochs      :{args.epochs}")
print(f"LR X_s      :{args.lr_X}")
print(f"Learn A_s   :{args.learn_A}")
print(f"LR A_s      :{lr_A}")
print(f"Conden ratio:{args.cond_ratio}")
print(f"Ridge       :{args.ridge}")
print(f"Kernel      :{args.kernel}")
print(f"k           :{args.k}")
# print(f"K           :{args.k}")
print(f"L           :{args.L}")
print(f"Seed        :{args.seed}")

# torch.autograd.set_detect_anomaly(True) 
def train(G_t, G_s, y_t, y_s, A_t, A_s, Alpha, loss_fn, epoch, learnA):

    pred, acc  = KRR.forward( G_t, G_s, y_t, y_s, A_t, A_s, Alpha, epoch,train=True, learnA=learnA)
    pred      = pred.to(torch.float32)
    y_t       = y_t.to(torch.float32)
    loss = loss_fn(pred, y_t).to(torch.float32)


    # with torch.autograd.detect_anomaly():
    optimizer_X.zero_grad()
    if learnA:
        optimizer_A.zero_grad()

    loss.backward()

    optimizer_X.step()
    if learnA:
        optimizer_A.step()

    # A_s.data.clamp_(min=0)
    loss = loss.item()

    # print(f"Training loss: {loss:>6f} ", end = ' ')
    return x_s, y_s, Alpha, loss, acc*100, pred


def test(G_t, G_s, y_t, y_s, A_t, A_s, Alpha, loss_fn, learnA):
    
    test_loss = 0
    with torch.no_grad():
        pred,acc    = KRR.forward( G_t, G_s, y_t, y_s, A_t, A_s, Alpha, train=False, learnA=learnA)
        test_loss  += loss_fn(pred, y_t).item()

    # print(f"Test Acc: {(100*acc):>0.2f}%, Avg loss: {test_loss:>6f}",end = '\n')
    return test_loss, acc*100

Time = torch.zeros(args.epochs,args.iter).to(device)
Acc = torch.zeros(args.epochs,args.iter).to(device)
for iter in range(args.iter):
    # print('--------------------------------------------------')
    print('The '+str(iter+1)+'th Iteration:')
    print('--------------------------------------------------')

    x_s = torch.rand(Cond_size, dim).to(device)
    y_s = torch.rand(Cond_size, n_class).to(device)
    A_s = torch.eye(Cond_size).to(x_s.device)

    x_s.requires_grad = True
    y_s.requires_grad = True
    

    Alpha = torch.rand(Cond_size, Cond_size).to(x_s.device) # to be optmized parameter of the BinConcrete distribution
    Alpha = Alpha * 2

    optimizer_X   = torch.optim.Adam([x_s,y_s], lr=args.lr_X)
    
    if args.learn_A:
        Alpha.requires_grad = True
        optimizer_A = torch.optim.Adam([Alpha], lr=lr_A)

    T = 0
    for epoch in range(args.epochs):
        # print(f"Epoch {iter+1}-{epoch+1}", end=" ")

        a = time.time()

        x_s, y_s, Alpha, training_loss, training_acc, pred = train(x_train, x_s, y_train_one_hot, y_s, E_train,A_s, Alpha,  MSEloss, epoch, args.learn_A)

        b = time.time()
        T = T + b-a
        Time[epoch,iter] = T

        AA = x_s.detach()
        BB = y_s.detach()
        CC = Alpha.detach()

        test_loss, test_acc = test(x_test, AA, y_test_one_hot, BB, E_test, A_s, CC,  MSEloss, args.learn_A)
        Acc[epoch,iter] = test_acc


# adj = Alpha.detach()
# adj = Alpha / ( 1 + Alpha )
# threshold = 0.5
# adj[adj> threshold] = 1
# adj[adj<= threshold] = 0
# adj = torch.clamp(adj+torch.eye(adj.shape[0]).to(adj.device), min=0, max=1)




Time_mean = torch.mean(Time, dim=1).cpu()

Acc_mean,Acc_std = torch.mean(Acc, dim=1).cpu(),torch.std(Acc, dim=1).cpu()
max_acc, max_ind = torch.max(Acc_mean, dim=0)

epochs = torch.tensor(range(args.epochs)) + 1
Time_Acc = torch.cat((epochs.unsqueeze(1),Acc_mean.unsqueeze(1),Acc_std.unsqueeze(1), Time_mean.unsqueeze(1)),dim=1)

np.set_printoptions(suppress=True)
print(np.array(Time_Acc.cpu()))

# print(sum(adj))

print("The best results: ")
print(f'Epoch: {max_ind + 1}', end = ' ')
print(' {:.2f}±{:.2f}'.format(max_acc, Acc_std[max_ind]), end = ' ')
print(f" {Time_mean[max_ind]:.3f}s")
print("--------------- Train Done! ----------------")


# print(args.save)

if args.save:
    adj = Alpha.detach()
    adj = Alpha / ( 1 + Alpha )
    threshold = 0.5
    adj[adj> threshold] = 1
    adj[adj<= threshold] = 0
    torch.save(x_s, 'save/'+args.dataset+'_x_s_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learn_A)+'.pt')
    torch.save(y_s, 'save/'+args.dataset+'_y_s_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learn_A)+'.pt')
    torch.save(adj, 'save/'+args.dataset+'_A_s_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learn_A)+'.pt')
    pred,_ = KRR.forward( x_s, x_s, y_s, y_s, Alpha, adj, adj,train=False)
    torch.save(pred,'save/'+args.dataset+'_pred_'+args.kernel+'_'+str(args.cond_ratio)+'_learnA_'+str(args.learn_A)+'.pt')
    print("--------------- Save Done! ----------------")