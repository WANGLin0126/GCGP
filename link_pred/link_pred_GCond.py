"""
using the condensed graph from GCond method to training a link prediction model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling, contains_self_loops, remove_self_loops
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
parser = argparse.ArgumentParser(description='Link Prediction with GCond')
parser.add_argument('--dataset', type=str, default="citeseer", help='dataset name [cora, citeseer]')
parser.add_argument('--cond_ratio', type=str, default="0.25", help='condensed graph ratio  [0.25, 0.5, 1.0]')
parser.add_argument('--seed', type=str, default="0", help='seed  [0, 1, 2, 3, 4]')
args = parser.parse_args()




try:
    dataset = Planetoid(root='./dataset', name=args.dataset)
    data = dataset[0].to(device)
    
    # 边划分并保存（保持与之前实验一致）
    data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
    torch.save({
        'train_pos_edge_index': data.train_pos_edge_index,
        'val_pos_edge_index': data.val_pos_edge_index,
        'val_neg_edge_index': data.val_neg_edge_index,
        'test_pos_edge_index': data.test_pos_edge_index,
        'test_neg_edge_index': data.test_neg_edge_index
    }, 'dataset_split.pt')
    print("数据划分完成")
except Exception as e:
    raise RuntimeError(f"加载数据集失败: {str(e)}")

# 2. 加载带有连续值的自定义训练数据
def load_continuous_train_data(x_path='x.pt', A_path='A.pt'):
    """ 加载包含连续边权重的训练数据 """
    try:
        # 加载数据
        x = torch.load(x_path).to(device)
        A = torch.load(A_path).to(device)
        
        # 验证数据格式
        assert x.dim() == 2 and A.dim() == 2
        assert x.size(0) == A.size(0), "节点数不匹配"
        
        # 提取非零边及其权重
        row, col = torch.where(A != 0)
        edge_index = torch.stack([row, col], dim=0).long()
        edge_weight = A[row, col]  # 连续值
        
        # 移除自环边
        if contains_self_loops(edge_index):
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    
    except Exception as e:
        raise RuntimeError(f"连续数据加载失败: {str(e)}")

# 3. 加载自定义连续数据
try:
    train_data = load_continuous_train_data(
        x_path=f"./saved_GCond/feat_{args.dataset}_{args.cond_ratio}_{args.seed}.pt",
        A_path=f"./saved_GCond/adj_{args.dataset}_{args.cond_ratio}_{args.seed}.pt"
    )
    print(f"连续数据加载成功: 节点数={train_data.num_nodes}, 边数={train_data.edge_index.size(1)}")
except Exception as e:
    print(e)
    exit(1)

# 4. 连续值预测模型
class ContinuousLinkPrediction(nn.Module):
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
        return (z[src] * z[dst]).sum(dim=-1)  # 输出连续值
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

# 5. 初始化模型
model = ContinuousLinkPrediction(
    in_channels=train_data.num_features,
    hidden_channels=128,
    out_channels=64
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"连续值模型初始化完成")

# 6. 训练函数（回归任务）
def train_continuous():
    model.train()
    optimizer.zero_grad()
    
    try:
        # 使用连续边权重作为目标值
        pred = model(train_data.x, train_data.edge_index)
        loss = F.mse_loss(pred, train_data.edge_weight)
        
        loss.backward()
        optimizer.step()
        return loss.item()
    except Exception as e:
        print(f"连续值训练错误: {str(e)}")
        exit(1)

# 7. 兼容性评估函数（保持与之前相同）
@torch.no_grad()
def test(split='test'):
    model.eval()
    try:
        # 加载预划分的边
        split_data = torch.load('dataset_split.pt', map_location=device)
        pos_edge_index = split_data[f'{split}_pos_edge_index']
        neg_edge_index = split_data[f'{split}_neg_edge_index']
        
        # 将连续预测值转换为概率
        pos_pred = torch.sigmoid(model(data.x, pos_edge_index))
        neg_pred = torch.sigmoid(model(data.x, neg_edge_index))
        
        # 计算 AUC
        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
        y_score = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        return roc_auc_score(y_true, y_score)
    
    except Exception as e:
        print(f"评估错误 ({split}): {str(e)}")
        return 0.0

# 8. 训练循环
best_auc = 0
try:
    for epoch in range(1, 301):
        loss = train_continuous()
        
        # 保持验证频率与之前一致
        if epoch % 20 == 0:
            val_auc = test('val')
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # 保存最佳模型
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'best_continuous_model.pth')
                # print(f"保存最佳连续模型，Val AUC: {val_auc:.4f}")
                
except KeyboardInterrupt:
    print("训练被用户中断")

# 9. 最终测试
model.load_state_dict(torch.load('best_continuous_model.pth'))
test_auc = test('test')
print("-"* 50)
print(args.dataset, end =" ")
print('Condensed Ratio:', args.cond_ratio)
print(f'Final Test AUC: {test_auc:.4f}')
print("-"* 50)