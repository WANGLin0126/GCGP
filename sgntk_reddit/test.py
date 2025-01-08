from torch_geometric.datasets import Flickr as DataSet
# import inspect

# Dataset       = Flickr(root="./datasets/Flickr")
root = "./datasets/Flickr"
dataset = DataSet(root=root, transform=None, pre_transform=None)

n, dim = dataset[0].x.shape

print(n, dim)