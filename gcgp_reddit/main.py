

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from torch.nn import functional as F
from gpr import GaussianProcessRegression
from sgtk import SimplifyingGraphTangentKernel
from sgnk import SimplifyingGraphNeuralKernel
from ntk import NeuralTangentKernel
from sntk import StructureBasedNTK
from utils import update_E
import argparse
import numpy as np
import random
import time
from utils import edge_ind_to_sparse_adj
from FlickrDataloader import FlickrDataLoader
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="Reddit", help='name of dataset [Reddit]')
parser.add_argument('--cond_size', type=int, default=77, help='condensation size)[77, 153, 307]')
parser.add_argument('--ridge', type=float, default=1e-3, help='ridge parameter of GPR (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 100)')
parser.add_argument('--lr_X', type=float, default=5e-3, help='learning rate (default: 0.005)')
parser.add_argument('--lr_A', type=float, default=5e-3, help='learning rate (default: 0.005)')
parser.add_argument('--K', type=int, default=0, help='number of aggr in SGNK (default: 2)')
parser.add_argument('--k', type=int, default=0, help='number of aggr in SGNK (default: 2)')
parser.add_argument('--L', type=int, default=2, help='the number of layers after each aggr (default: 2)')
parser.add_argument('--learn_A', type=int, default=0, help='whether to learn the adjacency matrix')
parser.add_argument('--norm', type=int, default=0, help='whether to normalize the features')
parser.add_argument('--set_seed', type=bool, default=True, help='setup the random seed (default: False)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--batch_size', type=int, default=8000, help='batch size (default: 4000)')
parser.add_argument('--accumulate_steps', type=int, default=8, help='accumulate steps (default: 10)')
parser.add_argument('--save', type=bool, default=False, help='save the results (default: False)')
parser.add_argument('--iterations', type=int, default=2, help='number of iterations of the whole experiments (default: 10)')
parser.add_argument('--kernel', type=str, default='SGNK', help='kernel method [SGNK] (default: SGNK)')
parser.add_argument('--num_hops', type=int, default=0, help='number of the hops when sampling the training batches (default: 0)')
args = parser.parse_args()

args.K = args.k

if args.learn_A:
    args.lr_A = args.lr_X
    
# torch.autograd.set_detect_anomaly(True) 
def train(G_t, G_s, y_t, y_s, A_t, A_s,Alpha, loss_fn, accumulate_steps, i, epoch, learnA, norm):
    pred, correct,loss_sparse = GPR.forward( G_t, G_s, y_t, y_s, A_t, A_s, Alpha,train=1, epoch=epoch, learnA = learnA, norm = norm)

    pred      = pred.to(torch.float32)
    y_t       = y_t.to(torch.float32)
    loss      = loss_fn(pred, y_t)
    loss      = loss + loss_sparse
    loss      = loss.to(torch.float32)

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

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
        A_s.data.clamp_(min=0, max=1)
    loss = loss.item()
    return x_s, y_s, loss, correct

def test(G_t, G_s, y_t, y_s, A_t, A_s, Alpha, loss_fn, learnA):
    size               = len(y_t)
    test_loss, correct = 0, 0
    with torch.no_grad():
        pred,_,_      = GPR.forward( G_t, G_s, y_t, y_s, A_t, A_s ,Alpha, train=0, epoch=0, learnA = learnA)
        test_loss  += loss_fn(pred, y_t).item()
        correct    += (pred.argmax(1) == y_t.argmax(1)).type(torch.float).sum().item()
    return test_loss, correct


train_loader = FlickrDataLoader(name = args.dataset, split='train', batch_size=args.batch_size, split_method='kmeans', device=device, k =args.k, num_hops=args.num_hops)
test_loader  = FlickrDataLoader(name = args.dataset, split='test', batch_size=10000, split_method='kmeans', device=device, k =args.k, num_hops=0)
TRAIN_K,n_train,n_class, dim, n  = train_loader.properties()
test_k,n_test,_,_,_              = test_loader.properties()

# TRAIN_K = train_loader.train_loader.__len__()
# train_loader.split_batch()
# test_loader.split_batch()

idx_s      = torch.tensor(range(round(args.cond_size)))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    setup_seed(args.seed)


def dot_product(x, y, A1, A2):
    return torch.matmul(x,y.T)

SGNK       = SimplifyingGraphNeuralKernel(L=args.L).to(device)
SGTK       = SimplifyingGraphTangentKernel(K=args.K, L=args.L).to(device)
SNTK       = StructureBasedNTK(K=args.K, L=args.L).to(device)
NTK         = NeuralTangentKernel( L = 2 ).to(device)


ridge      = torch.tensor(args.ridge).to(device)

if args.kernel == 'SGTK':
    kernel     = SGTK.nodes_gram
elif args.kernel == 'SGNK':
    kernel     = SGNK.nodes_gram
elif args.kernel == 'SNTK':
    kernel     = SNTK.nodes_gram
elif args.kernel == "dot_product":
    kernel      =  dot_product
elif args.kernel == "NTK":
    kernel      =  NTK.nodes_gram

GPR        = GaussianProcessRegression(kernel,ridge,args.K).to(device)



print(f"Dataset       :{args.dataset}")
print(f"Conden size   :{args.cond_size}")
print(f"Ridge         :{args.ridge}")
# print(f"Number        :{n}")
# print(f"Training Set  :{n_train}")
# print(f"Testing Set   :{n_test}")
# print(f"Classes       :{n_class}")
# print(f"Dim           :{dim}")
print(f"Epochs        :{args.epochs}")
print(f"Learn A       :{args.learn_A}")
print(f"Norm          :{args.norm}")
print(f"Aggr (K)      :{args.K}")
print(f"L             :{args.L}")
print(f"LR X          :{args.lr_X}")
print(f"LR A          :{args.lr_A}")
print(f"Batches       :{TRAIN_K}")
print(f"Kernel        :{args.kernel}")













torch.cuda.reset_peak_memory_stats()

Time = torch.zeros(args.epochs,args.iterations)
results = torch.zeros(args.epochs,args.iterations)
for iter in range(args.iterations):
    print(f"The  {iter+1}-th iteration")
    print("---------------------------------------------")
    x_s = torch.rand(round(args.cond_size), dim)
    y_s = torch.rand(round(args.cond_size), n_class)
    A_s = torch.eye(args.cond_size).to(x_s.device)
    # A_s = torch.rand(args.cond_size, args.cond_size).to(x_s.device)

    MSEloss = nn.MSELoss().to(device)
    # MSEloss = nn.CrossEntropyLoss().to(device)

    idx_s   = idx_s.to(device)
    x_s     = x_s.to(device)
    y_s     = y_s.to(device)
    A_s     = A_s.to(device)
    x_s.requires_grad = True
    y_s.requires_grad = True
    A_s.requires_grad = True

    optimizer_X = torch.optim.Adam([x_s,y_s], lr=args.lr_X)
    Alpha = torch.rand(args.cond_size, args.cond_size).to(device)
    Alpha = Alpha * 2 
    if args.learn_A:
        Alpha.requires_grad = True
        optimizer_A = torch.optim.Adam([Alpha], lr=args.lr_A)

    max_test_acc = 0
    start = time.time()

    T = 0
    for t in range(args.epochs):
        a = time.time()
        print(f"Epoch {t+1}", end=" ")
        train_loss, test_lossi = torch.zeros(TRAIN_K),  torch.zeros(test_k)
        train_correct_all, test_correct_all = 0, 0
        i = 0
        # torch.cuda.empty_cache()
        for data in train_loader.loader():
        #     # x_train, label, sub_A_t  = train_loader.get_batch(i)
            x_train = data.x 
            label   =  data.y
            edge_index = data.edge_index
            sub_A_t = edge_ind_to_sparse_adj(edge_index)

            y_train_one_hot          = F.one_hot(label.reshape(-1), n_class)

            # x_train = x_train#.to(device)
            # y_train_one_hot = y_train_one_hot#.to(device)
            # sub_A_t = sub_A_t#.to(device)

            _, _, training_loss, train_correct = train(x_train, x_s, y_train_one_hot, y_s, sub_A_t, A_s, Alpha, MSEloss, args.accumulate_steps, i, t, args.learn_A, args.norm)

            train_correct_all = train_correct_all + train_correct
            train_loss[i]     = training_loss
            i = i + 1

        b = time.time()
        T = T + b-a
        Time[t,iter] = T
        training_loss_avg = torch.mean(train_loss)
        training_acc_avg = (train_correct_all / n_train) * 100
        # test_a = time.time()

        # AA = x_s.detach()
        # BB = y_s.detach()
        # CC = A_s.detach()

        # adj = CC
        # adj = torch.sigmoid(adj)
        # adj[adj> 0.5] = 1
        # adj[adj<= 0.5] = 0
        # print(sum(adj))

        if t >= 0:
            j = 0

            AA = x_s.detach()
            BB = y_s.detach()
            CC = Alpha.detach()
            # for j in range(test_k):
            for data in test_loader.loader():
                
                x_test = data.x 
                test_label   =  data.y
                edge_index = data.edge_index
                sub_A_test = edge_ind_to_sparse_adj(edge_index)

                # x_test, test_label, sub_A_test  = test_loader.get_batch(j)
                y_test_one_hot       = F.one_hot(test_label.reshape(-1), n_class).to(torch.float32)

                x_test = x_test.to(device)
                y_test_one_hot = y_test_one_hot.to(device)
                sub_A_test = sub_A_test.to(device)

                test_loss, test_correct = test(x_test, AA, y_test_one_hot, BB, sub_A_test, A_s, CC,  MSEloss, args.learn_A)

                test_correct_all = test_correct_all + test_correct
                test_lossi[j] = test_loss
                j = j + 1

            test_loss_avg = torch.mean(test_lossi)
            test_acc      = (test_correct_all / n_test) * 100
            print(f"Test Acc: {(test_acc):>0.4f}%, Test loss: {test_loss_avg:>0.6f}",end = '\n')
            results[t,iter] = test_acc

        test_b = time.time()

    end = time.time()
    # print('Running time: %s Seconds'%(end-start))

    print("---------------------------------------------")

# print(f"Dataset       :{args.dataset}")
# print(f"Conden size   :{args.cond_size}")
# print(f"Ridge         :{args.ridge}")
# # print(f"Number        :{n}")
# # print(f"Training Set  :{n_train}")
# # print(f"Testing Set   :{n_test}")
# # print(f"Classes       :{n_class}")
# # print(f"Dim           :{dim}")
# print(f"Epochs        :{args.epochs}")
# print(f"Learn A       :{args.learn_A}")
# print(f"Norm          :{args.norm}")
# print(f"Aggr (K)      :{args.K}")
# print(f"L             :{args.L}")
# print(f"LR X          :{args.lr_X}")
# print(f"LR A          :{args.lr_A}")
# print(f"Batches       :{TRAIN_K}")
# print(f"Kernel        :{args.kernel}")

print("---------------------------------------------")
epochs = torch.tensor(range(args.epochs)) + 1
Acc_mean,Acc_std = torch.mean(results, dim=1),torch.std(results, dim=1)
max_mean, max_mean_index = torch.max(Acc_mean, dim=0)
Time_mean = torch.mean(Time, dim=1)
Time_Acc = torch.cat((epochs.unsqueeze(1),Acc_mean.unsqueeze(1), Acc_std.unsqueeze(1), Time_mean.unsqueeze(1)),dim=1)

np.set_printoptions(suppress=True, linewidth=300, threshold=np.inf)
print(np.array(Time_Acc.cpu()))
print(f'Best Result: Epoch   Acc  Std.') 
print(f'                {max_mean_index+1} {max_mean.item():>0.2f}  {Acc_std[max_mean_index].item():>0.2f}')
print("--------------- Train Done! ----------------")
# load file Time_Acc.pt
# Time_Acc = torch.load('Time_Acc.pt')
# print(np.array(Time_Acc.cpu()))

# pred,_,_ = GPR.forward( x_s, x_s, y_s, y_s, A_s, A_s,train=False)

if args.save:
    adj = Alpha.detach()
    adj = Alpha / ( 1 + Alpha )
    threshold = 0.5
    adj[adj> threshold] = 1
    adj[adj<= threshold] = 0
    torch.save(x_s, 'save/'+args.dataset+'_x_s_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    torch.save(y_s, 'save/'+args.dataset+'_y_s_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    torch.save(adj, 'save/'+args.dataset+'_A_s_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    pred,_ = GPR.forward( x_s, x_s, y_s, y_s, adj, adj,train=False)
    torch.save(pred,'save/'+args.dataset+'_pred_'+str(args.cond_size)+'_learnA_'+str(args.learn_A)+'.pt')
    print("--------------- Save Done! ----------------")