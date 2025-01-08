import torch
from OgbLoader import OgbNodeDataLoader

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(torch.cuda.is_available())
print(torch.cuda.device_count())


OGB = OgbNodeDataLoader(dataset_name='ogbn-arxiv', train_batch_size=3000, test_batch_size = 10000, aggr = 2, num_hops = 1, device=device)

n =0
for data in OGB.train_loader():
    x =  data.x
    edge_index = data.edge_index
    y = data.y
    batch_size = x.shape[0]
    n = n + batch_size
    print(x.shape[0])
print(n)


n =0
for data in OGB.test_loader():
    x =  data.x
    edge_index = data.edge_index
    y = data.y
    batch_size = x.shape[0]
    n = n + batch_size
    print(x.shape[0])
print(n)


# 创建两个稀疏矩阵A和B
# 这里使用的是COO（坐标）格式
A_indices = torch.tensor([[0, 0], [1, 1], [2, 2]])
A_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
A_size = torch.Size([3, 3])

A = torch.sparse_coo_tensor(indices=A_indices.t(), values=A_values, size=A_size)

B_indices = torch.tensor([[0, 0], [2, 1]])
B_values = torch.tensor([4.0, 5.0], dtype=torch.float32)
B_size = torch.Size([3, 2])
B = torch.sparse_coo_tensor(indices=B_indices.t(), values=B_values, size=B_size)

# 进行稀疏矩阵乘法
C = torch.sparse.mm(A, B)

# 将结果转换为密集矩阵进行打印
C_dense = C.to_dense()
print(C_dense)

print(torch.__version__)



import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

# 假设我们有一个简单的图
edge_index = torch.tensor([
    [0, 1, 1, 2, 3, 4],
    [1, 0, 2, 1, 4, 3]
], dtype=torch.long)

# 创建图数据
data = Data(edge_index=edge_index)

# 给定节点和 k-hop 的参数
node_idx = 0  # 要查找的节点
k = 2  # k-hop

# 使用 k_hop_subgraph 函数查找 k-hop 邻居
subset, edge_index_k_hop, mapping, edge_mask = k_hop_subgraph(node_idx=node_idx, num_hops=k, edge_index=data.edge_index, relabel_nodes=True)

print("k-hop 邻居节点索引:", subset)
print("k-hop 子图的边索引:", edge_index_k_hop)