import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges, contains_self_loops, remove_self_loops
from sklearn.metrics import roc_auc_score

# 设置全局随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# 1. 加载并预处理 Cora 数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    cora_dataset = Planetoid(root='./dataset', name='Cora')
    cora_data = cora_dataset[0].to(device)
    
    # 边划分并保存
    cora_data = train_test_split_edges(cora_data, val_ratio=0.05, test_ratio=0.1)
    torch.save({
        'train_pos_edge_index': cora_data.train_pos_edge_index,
        'val_pos_edge_index': cora_data.val_pos_edge_index,
        'val_neg_edge_index': cora_data.val_neg_edge_index,
        'test_pos_edge_index': cora_data.test_pos_edge_index,
        'test_neg_edge_index': cora_data.test_neg_edge_index
    }, 'cora_split.pt')
    print("Cora 数据划分完成")
except Exception as e:
    raise RuntimeError(f"加载 Cora 数据集失败: {str(e)}")

# 2. 自定义数据加载函数
def load_custom_train_data(x_path='x.pt', A_path='A.pt'):
    """ 从 .pt 文件加载自定义训练数据 """
    try:
        # 检查文件存在性
        if not all(os.path.exists(p) for p in [x_path, A_path]):
            raise FileNotFoundError("部分数据文件不存在")
            
        # 加载数据并检查格式
        x = torch.load(x_path).to(device)
        A = torch.load(A_path).to(device)
        assert x.dim() == 2, "节点特征矩阵 x 必须是二维张量"
        assert A.dim() == 2, "邻接矩阵 A 必须是二维张量"
        assert x.size(0) == A.size(0), "节点数和邻接矩阵维度不匹配"

        # 转换邻接矩阵为边索引
        edge_index = A.nonzero(as_tuple=False).t().contiguous().to(torch.long)
        
        # 移除自环边
        if contains_self_loops(edge_index):
            edge_index, _ = remove_self_loops(edge_index)
            
        return Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    
    except Exception as e:
        raise RuntimeError(f"自定义数据加载失败: {str(e)}")

# 3. 加载自定义训练数据
try:
    train_data = load_custom_train_data(
        x_path="./saved_GCGP/Cora_x_s_SGNK_0.5_learnA_1.pt", 
        A_path="./saved_GCGP/Cora_A_s_SGNK_0.5_learnA_1.pt"
    )
    print(f"自定义数据加载成功: 节点数={train_data.num_nodes}, 边数={train_data.edge_index.size(1)}")
except Exception as e:
    print(e)
    exit(1)

# 4. 修改后的模型定义
class LinkPredictionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index  # 返回嵌入和边索引
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)
    
    def forward(self, x, edge_index):
        z, edge_index = self.encode(x, edge_index)
        return z, edge_index

# 5. 初始化模型
model = LinkPredictionModel(
    in_channels=train_data.num_features,
    hidden_channels=128,
    out_channels=64
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
print(f"模型初始化完成: {model}")

# 6. 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    
    try:
        x, edge_index = train_data.x, train_data.edge_index
        
        # 负采样
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=edge_index.size(1)
        ).to(device)
        
        # 前向传播并解码
        z, _ = model(x, edge_index)  # 获取节点嵌入
        pos_out = model.decode(z, edge_index)
        neg_out = model.decode(z, neg_edge_index)
        
        # 损失计算
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        return loss.item()
    
    except Exception as e:
        print(f"训练错误: {str(e)}")
        exit(1)

# 7. 增强型评估函数
@torch.no_grad()
def test(split='test'):
    model.eval()
    try:
        # 加载划分数据
        split_data = torch.load('cora_split.pt', map_location=device)
        pos_edge_index = split_data[f'{split}_pos_edge_index']
        neg_edge_index = split_data[f'{split}_neg_edge_index']
        
        # 前向传播
        z, _ = model(cora_data.x, pos_edge_index)
        
        # 解码预测
        pos_pred = model.decode(z, pos_edge_index)
        neg_pred = model.decode(z, neg_edge_index)
        
        # 计算指标
        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
        y_score = torch.cat([pos_pred, neg_pred]).sigmoid().cpu().numpy()
        return roc_auc_score(y_true, y_score)
    
    except Exception as e:
        print(f"评估错误 ({split}): {str(e)}")
        return 0.0

# 8. 训练循环
best_auc = 0
try:
    for epoch in range(1, 501):
        loss = train()
        
        # 每20个epoch验证一次
        if epoch % 20 == 0:
            val_auc = test('val')
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # 保存最佳模型
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"保存最佳模型，Val AUC: {val_auc:.4f}")
                
except KeyboardInterrupt:
    print("训练被用户中断")

# 9. 最终测试
model.load_state_dict(torch.load('best_model.pth'))
test_auc = test('test')
print(f'Final Test AUC: {test_auc:.4f}')

# # 10. 设备验证
# print("\n设备一致性检查:")
# print(f"- 模型参数设备: {next(model.parameters()).device}")
# print(f"- 训练数据设备: {train_data.x.device}")
# print(f"- Cora 数据设备: {cora_data.x.device}")