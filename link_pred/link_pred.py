"""
using the original training set to training a link prediction model

"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score
import argparse

# 设置全局随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# 1. 加载并预处理数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 初始化 参数
parser = argparse.ArgumentParser(description='Link Prediction with GCGP')
parser.add_argument('--dataset', type=str, default="Cora", help='dataset name')
args = parser.parse_args()


try:
    if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./dataset', name=args.dataset)
    elif args.dataset in ['Photo', 'Computers']:
        dataset = Amazon(root='./dataset', name=args.dataset)
    
    data = dataset[0].to(device)
    
    # 备份原始边索引
    original_edge_index = data.edge_index.clone()
    
    # 边划分并保存（保持与之前实验一致）
    data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
    torch.save({
        'train_pos_edge_index': data.train_pos_edge_index,
        'val_pos_edge_index': data.val_pos_edge_index,
        'val_neg_edge_index': data.val_neg_edge_index,
        'test_pos_edge_index': data.test_pos_edge_index,
        'test_neg_edge_index': data.test_neg_edge_index
    }, 'dataset_split.pt')
    
    # 恢复原始边索引
    data.edge_index = original_edge_index
    print("数据划分完成")
except Exception as e:
    raise RuntimeError(f"加载数据集失败: {str(e)}")

# 2. 使用原始训练节点生成训练边
def generate_train_edges(data):
    """ 从原始训练节点生成训练边 """
    try:
        # 获取训练节点mask
        train_mask = data.train_mask
        
        # 提取训练节点索引
        train_nodes = torch.where(train_mask)[0]
        
        # 筛选训练节点之间的边
        src, dst = data.edge_index
        train_edge_mask = torch.isin(src, train_nodes) & torch.isin(dst, train_nodes)
        train_pos_edge_index = data.edge_index[:, train_edge_mask]
        
        return Data(
            x=data.x,
            edge_index=train_pos_edge_index,
            num_nodes=data.num_nodes
        )
    except Exception as e:
        raise RuntimeError(f"生成训练边失败: {str(e)}")

# 3. 生成训练数据
try:
    train_data = generate_train_edges(data)
    print(f"原始训练边生成成功: 边数={train_data.edge_index.size(1)}")
except Exception as e:
    print(e)
    exit(1)

# 4. 模型定义（保持不变）
class LinkPredictionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)
    
    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

# 5. 初始化模型
model = LinkPredictionModel(
    in_channels=dataset.num_features,
    hidden_channels=128,
    out_channels=64
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"模型初始化完成")

# 6. 训练函数（保持不变）
def train():
    model.train()
    optimizer.zero_grad()
    
    try:
        x = train_data.x.to(device)
        edge_index = train_data.edge_index.to(device)
        
        # 负采样
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=edge_index.size(1)
        ).to(device)
        
        # 前向传播
        z = model(x, edge_index)
        pos_out = model.decode(z, edge_index)
        neg_out = model.decode(z, neg_edge_index)
        
        # 损失计算
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        return loss.item()
    
    except Exception as e:
        print(f"训练错误: {str(e)}")
        exit(1)

# 7. 评估函数（保持不变）
@torch.no_grad()
def test(split='test'):
    model.eval()
    try:
        # 加载预划分数据
        split_data = torch.load('dataset_split.pt', map_location=device)
        pos_edge_index = split_data[f'{split}_pos_edge_index']
        neg_edge_index = split_data[f'{split}_neg_edge_index']
        
        # 预测
        z = model(data.x, pos_edge_index)
        pos_pred = model.decode(z, pos_edge_index)
        neg_pred = model.decode(z, neg_edge_index)
        
        # 计算 AUC
        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
        y_score = torch.cat([pos_pred, neg_pred]).sigmoid().cpu().numpy()
        return roc_auc_score(y_true, y_score)
    
    except Exception as e:
        print(f"评估错误 ({split}): {str(e)}")
        return 0.0

# 8. 训练循环（保持不变）
best_auc = 0
try:
    for epoch in range(1, 301):
        loss = train()
        
        if epoch % 20 == 0:
            val_auc = test('val')
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'best_original_model.pth')
                
except KeyboardInterrupt:
    print("训练被用户中断")

# 9. 最终测试
model.load_state_dict(torch.load('best_original_model.pth'))
test_auc = test('test')
print('-'*50)
print(args.dataset)
print(f'Final Test AUC: {test_auc:.4f}')
print('-'*50)