import torch
from ogb.nodeproppred import dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from utils import edge_ind_to_sparse_adj
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch import nn
from torch.nn import functional as F
from gpr import GaussianProcessRegression
from sntk import StructureBasedNeuralTangentKernel
from sgnk import SimplifyingGraphNeuralKernel
from ntk import NeuralTangentKernel
from lightgntk import LightGraphNeuralTangentKernel
from utils import sub_G, sub_A_list, update_E, sub_E
import argparse
import numpy as np
import random
import time
from OGBDataloader import OgbDataLoader
from OgbLoader import OgbNodeDataLoader


def train(G_t, G_s, y_t, y_s, A_t, A_s, Alpha, loss_fn, accumulate_steps,i, epoch, learnA, norm):
    pred, correct = GPR.forward( G_t, G_s, y_t, y_s, A_t, A_s, Alpha, epoch, train = 1, learnA=learnA, norm = norm)

    if sum(sum(A_s<0)):
        raise ValueError(f"Training loss is {A_s}")

    pred      = pred.to(torch.float32)
    y_t       = y_t.to(torch.float32)
    loss      = loss_fn(pred, y_t)
    loss      = loss.to(torch.float32)

    # print(torch.isnan(torch.any(pred)))

    loss = loss/accumulate_steps
    loss.backward()

    if (i+1) % accumulate_steps == 0:

        if learnA:
            optimizer_A.step()
            optimizer_A.zero_grad()

        optimizer_X.step()
        optimizer_X.zero_grad()

        A_s.data.clamp_(min=0, max=1)
    elif i == TRAIN_K - 1:
        if learnA:
            optimizer_A.step()
            optimizer_A.zero_grad()

        optimizer_X.step()
        optimizer_X.zero_grad()
        # A_s.data.clamp_(min=0, max=1)

    loss = loss.item()

    # if training_loss is nan or inf, raise error
    # if torch.isnan(torch.tensor(loss)):
    #     raise ValueError(f"Training loss is {loss}")

    return x_s, y_s, loss, correct, pred

def test(G_t, G_s, y_t, y_s, A_t, A_s, Alpha, loss_fn, learnA):
    size               = len(y_t)
    test_loss, correct = 0, 0
    with torch.no_grad():
        pred,_      = GPR.forward( G_t, G_s, y_t, y_s, A_t, A_s,Alpha, train = 0, learnA=learnA)
        test_loss  += loss_fn(pred, y_t).item()
        correct    += (pred.argmax(1) == y_t.argmax(1)).type(torch.float).sum().item()
    return test_loss, correct

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="ogbn-arxiv", help='name of dataset (default: ogbn-arxiv)')
parser.add_argument('--cond_size', type=int, default=90, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
parser.add_argument('--ridge', type=float, default=1e-3, help='parameter of GPR (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train (default: 100)')
parser.add_argument('--lr_X', type=float, default=5e-3, help='learning rate (default: 0.005)')
parser.add_argument('--lr_A', type=float, default=5e-3, help='learning rate (default: 0.005)')
# parser.add_argument('--k', type=int, default=0, help='the convolutiona times of the dataset (default: 2)')
parser.add_argument('--K', type=int, default=2, help='number of aggr in SGNK (default: 2)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 2)')
parser.add_argument('--learn_A', type=int, default=0, help='whether to learn the adjacency matrix')
parser.add_argument('--norm', type=int, default=0, help='whether to normalize the features')
parser.add_argument('--set_seed', type=bool, default=True, help='whether to set seed')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--iter', type=int, default=2, help='iteration times (default: 3)')
parser.add_argument('--kernel', type=str, default='LightGNTK', help='kernel method [SGNK] (default: SGNK)')
parser.add_argument('--split_method', type=str, default='random', help='split method of the test set [kmeans,none] (default: kmeans)')
parser.add_argument('--train_batch_size', type=int, default=5000, help='split method of the test set [kmeans,none] (default: kmeans)')
parser.add_argument('--save', type=int, default=0, help='whether to save the results')
args = parser.parse_args()



name = args.dataset

OGBloader = OgbNodeDataLoader(dataset_name=args.dataset, train_batch_size=args.train_batch_size, test_batch_size = 10000, aggr = args.K, num_hops = 1, device=device)
TRAIN_K, test_k, n_train, n_test, n_class, dim, n = OGBloader.properties()

# test_loader.split_batch()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if args.set_seed:
    setup_seed(args.seed)

SGTK       = StructureBasedNeuralTangentKernel( L=args.L).to(device)
SGNK       = SimplifyingGraphNeuralKernel(L=args.L).to(device)
NTK        = NeuralTangentKernel(L=args.L).to(device)
SNTK       = StructureBasedNeuralTangentKernel(K=args.K, L=args.L).to(device)
LightGNTK  = LightGraphNeuralTangentKernel(K=args.K).to(device)
ridge      = torch.tensor(args.ridge).to(device)


def dot_product(x, y, A1,A2):
    return torch.matmul(x, y.t())

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
elif args.kernel == "LightGNTK":
    kernel      =  LightGNTK.nodes_gram

GPR        = GaussianProcessRegression(kernel, ridge, K = args.K).to(device)

if args.learn_A:
    args.lr_A = args.lr_X


print(f"Dataset       :{args.dataset}")
print(f"Conden size   :{args.cond_size}")
print(f"Kernel        :{args.kernel}")
print(f"Ridge         :{args.ridge}")
print(f"Epochs        :{args.epochs}")
print(f"Learn A       :{args.learn_A}")
print(f"Norm          :{args.norm}")
print(f"Aggr (K)      :{args.K}")
print(f"L             :{args.L}")
print(f"LR X          :{args.lr_X}")
print(f"LR A          :{args.lr_A}")
print(f"Batches       :{TRAIN_K}")
print(f"Batch Size    :{args.train_batch_size}")
# print(f"Save          :{args.save}")


Time = torch.zeros(args.epochs,args.iter)
results = torch.zeros(args.epochs,args.iter)
for iter in range(args.iter):
    # print('--------------------------------------------------')
    print(f"The  {iter+1}-th iteration")
    print('--------------------------------------------------')


    idx_s  = torch.tensor(range(round(args.cond_size)))
    x_s = torch.rand(int(args.cond_size), dim)
    y_s = torch.rand(int(args.cond_size), n_class)
    A_s    = torch.eye(args.cond_size).to(x_s.device)


    MSEloss = nn.MSELoss().to(device)
    idx_s   = idx_s.to(device)
    x_s     = x_s.to(device)
    y_s     = y_s.to(device)
    A_s     = A_s.to(device)
    x_s.requires_grad = True
    y_s.requires_grad = True
    
    Alpha = torch.rand(int(args.cond_size), int(args.cond_size)).to(device) # to be optmized parameter of the BinConcrete distribution
    Alpha = Alpha * 2

    optimizer_X = torch.optim.Adam([x_s,y_s], lr=args.lr_X)
    if args.learn_A:
        Alpha.requires_grad = True
        optimizer_A = torch.optim.Adam([Alpha], lr=args.lr_A)

    T = 0
    for t in range(args.epochs):
        aa = time.time()
        print(f"Size {args.cond_size} Iter {iter+1} - {t+1}", end=" ")

        train_loss, test_lossi = torch.zeros(TRAIN_K),  torch.zeros(test_k)
        train_correct_all, test_correct_all = 0, 0
        a = time.time()
        i = 0
        # for data in loader.train_loader():
        for data in OGBloader.train_loader():
            x_train = data.x 
            label   =  data.y
            edge_index = data.edge_index
            sub_A_t = edge_ind_to_sparse_adj(edge_index)

            y_train_one_hot          = F.one_hot(label.reshape(-1), n_class)

            x_train = x_train.to(device)
            y_train_one_hot = y_train_one_hot.to(device)
            sub_A_t = sub_A_t.to(device)

            _, _, training_loss, train_correct, pred = train(x_train, x_s, y_train_one_hot, y_s, sub_A_t, A_s, Alpha, MSEloss, 10, i, t, args.learn_A, args.norm)


            train_correct_all = train_correct_all + train_correct
            train_loss[i]     = training_loss
            i = i + 1
        training_loss_avg = torch.mean(train_loss)
        training_acc_avg = (train_correct_all / n_train) * 100

        b = time.time()
        T = T + b - a
        Time[t,iter] = T

        # AA = x_s.detach()
        # BB = y_s.detach()
        # CC = A_s.detach()

        # adj = CC
        # adj = torch.sigmoid(adj)
        # threshold = 0.5
        # adj[adj> threshold] = 1
        # adj[adj<= threshold] = 0
        # print(sum(adj))


        if t >= 0:
            j = 0
            # for j in range(test_k):
            for data in OGBloader.test_loader():
                x_test, test_label, edge_index  = data.x, data.y, data.edge_index
                sub_A_test = edge_ind_to_sparse_adj(edge_index)

                # x_test, test_label, sub_A_test  = loader.get_test_batch(j)

                y_test_one_hot       = F.one_hot(test_label.reshape(-1), n_class)

                x_test = x_test.to(device)
                y_test_one_hot = y_test_one_hot.to(device)
                sub_A_test = sub_A_test.to(device)

                test_loss, test_correct = test(x_test, x_s, y_test_one_hot, y_s, sub_A_test, A_s, Alpha,  MSEloss, args.learn_A)

                test_correct_all = test_correct_all + test_correct
                test_lossi[j] = test_loss
                # j = j + 1
            bb = time.time()
            test_loss_avg = torch.mean(test_lossi)
            test_acc      = (test_correct_all / n_test) * 100
            print(f"Test Avg Acc: {(test_acc):>0.4f}%, Test Avg loss: {test_loss_avg:>0.6f}", end=" ")
            print(f"Time: {bb-aa:>0.4f}s",end='\n')
            results[t,iter] = test_acc

# print(f"Dataset       :{args.dataset}")
# print(f"Conden size   :{args.cond_size}")
# print(f"Ridge         :{args.ridge}")
# print(f"Kernel        :{args.kernel}")
# print(f"GC            :{args.k}")
# print(f"Aggr          :{args.K}")
# print(f"Layers        :{args.L}")
# print(f"Learning rate :{args.lr_X}")
# print(f"Learning rate :{args.lr_A}")
# print(f"Epochs        :{args.epochs}")

epochs = torch.tensor(range(args.epochs)) + 1
Acc_mean,Acc_std = torch.mean(results, dim=1),torch.std(results, dim=1)
max_mean, max_mean_index = torch.max(Acc_mean, dim=0)
Time_mean = torch.mean(Time, dim=1)
Time_Acc = torch.cat((epochs.unsqueeze(1), Acc_mean.unsqueeze(1), Acc_std.unsqueeze(1), Time_mean.unsqueeze(1)), dim=1)

np.set_printoptions(suppress=True)
print(np.array(Time_Acc.cpu()))
print(f'Best Result: Epoch   Acc  Std.')
print(f'                 {max_mean_index+1} {max_mean.item():>0.2f}  {Acc_std[max_mean_index].item():>0.2f}')
print("--------------- Train Done! ----------------")



# pred,_ = GPR.forward( x_s, x_s, y_s, y_s, A_s, A_s,train=False)

if args.save:
    adj = A_s.detach()
    adj = torch.sigmoid(adj)
    threshold = 0.5
    adj[adj> threshold] = 1
    adj[adj<= threshold] = 0
    torch.save(x_s, 'save/'+args.dataset+'_x_s_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    torch.save(y_s, 'save/'+args.dataset+'_y_s_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    torch.save(adj, 'save/'+args.dataset+'_A_s_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    pred,_ = GPR.forward( x_s, x_s, y_s, y_s, adj, adj,train=False)
    torch.save(pred,'save/'+args.dataset+'_pred_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    print("--------------- Save Done! ----------------")