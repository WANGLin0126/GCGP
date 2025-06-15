## 画出 arxiv 训练集的 类别样本数量分布


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
from utils import sub_G, sub_A_list, update_E, sub_E
import argparse
import numpy as np
import random
import time
from OGBDataloader import OgbDataLoader
from OgbLoader import OgbNodeDataLoader


device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
parser.add_argument('--kernel', type=str, default='SNTK', help='kernel method [SGNK] (default: SGNK)')
parser.add_argument('--split_method', type=str, default='random', help='split method of the test set [kmeans,none] (default: kmeans)')
parser.add_argument('--train_batch_size', type=int, default=5000, help='split method of the test set [kmeans,none] (default: kmeans)')
parser.add_argument('--save', type=int, default=0, help='whether to save the results')
args = parser.parse_args()



name = args.dataset




# # for data in loader.train_loader():
# for data in OGBloader.train_loader():
#     x_train = data.x 
#     label   =  data.y
    
    

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_label_distribution(dataloader):
    """ 统计标签分布 """
    label_counts = defaultdict(int)
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # 假设标签在batch的第二个位置，根据实际情况调整
            labels = batch.y  
            
            # 转换标签到numpy数组
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            
            # 统计当前batch
            unique, counts = np.unique(labels, return_counts=True)
            for cls, cnt in zip(unique, counts):
                label_counts[int(cls)] += int(cnt)
            
            total_samples += len(labels)

    return dict(sorted(label_counts.items()))

def plot_class_distribution(label_dict, save_path=None):
    """ 绘制带数值标注的柱状图 """
    classes = list(label_dict.keys())
    counts = list(label_dict.values())
    total = sum(counts)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='#1f77b4', width=0.8)
    # edgecolor='black'
    
    # 标注设置
    max_height = max(counts)
    offset = max_height * 0.02
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + offset,
                f'{count}\n({count/total:.1%})',
                ha='center', 
                va='bottom',
                fontsize=8)
    
    plt.xticks(classes, rotation=45, ha='right')
    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Ogbn-arxiv Training Set Class Distribution (Total: {total} samples)', fontsize=14)
    plt.ylim(0, max_height * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例 --------------------------------------------------
if __name__ == '__main__':
    # 假设已有训练集dataloader
    OGBloader = OgbNodeDataLoader(dataset_name=args.dataset, train_batch_size=args.train_batch_size, test_batch_size = 10000, aggr = args.K, num_hops = 0, device=device)

    
    # 1. 统计分布
    label_dist = analyze_label_distribution(OGBloader.train_loader())
    
    # 2. 打印统计结果
    print("Class Distribution:")
    for cls, cnt in label_dist.items():
        print(f"  Class {cls}: {cnt} samples")
    
    # 3. 可视化
    plot_class_distribution(label_dist, save_path='class_distribution.pdf')
