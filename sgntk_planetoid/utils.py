import torch
from torch_geometric.nn import MessagePassing
"""
Transform a single, large graph with n nodes to n subgraphs
"""

def find(idx,A,c=0):
    """
    find out the one-hop neighbors of nodes in given idx
    len(list) = len(idx)
    A must tensor
    """
    list = []
    for node in idx:
        neigh = torch.where(A[node]==1)[0]
        if c :
            list.append(neigh)
        else:
            for i in range(len(neigh)):
                list.append(neigh[i])
    if c:
        return list
    else:
        return torch.unique(torch.tensor(list))


def find_hop_idx(i,j,A):
    """
    find the index of the j-hop neighbors of node i
    """
    idx = [i,]
    for hop in range(j):
        idx = find(idx,A)
    return idx


def sub_G(A, hop):
    """
    A is the adjacency martrix of graph
    """
    n         = A.shape[0]
    neighbors = []
    for i in range(n):
        neighbor_i = torch.tensor(find_hop_idx(i,hop,A))
        neighbors.append(neighbor_i.to(A.device))
    return neighbors



def sub_A_list(neighbors, A):
    """
    output the adjacency matrix of subgraph
    """
    n        = A.shape[0]
    sub_A_list= []
    for node in range(n):

        n_neig   = len(neighbors[node])
        operator = torch.zeros([n,n_neig]).to(A.device)
        operator[neighbors[node],range(n_neig)] = 1
        sub_A = torch.matmul(torch.matmul(operator.t(),A),operator)
        sub_A_list.append(sub_A)

    return sub_A_list


def sub_A(idx, A):
    """
    output the adjacency matrix of subgraph of idx
    """
    n        = A.shape[0]
    n_neig   = len(idx)
    operator = torch.zeros([n,n_neig])
    operator[idx,range(n_neig)] = 1
    sub_A    = torch.matmul(torch.matmul(operator.t(),A),operator)

    return sub_A


def sub_E(idx, A):
    """
    output the sparse adjacency matrix of subgraph of idx
    """
    n        = A.shape[0]
    n_neig   = len(idx)
    operator = torch.zeros([n,n_neig])
    operator[idx,range(n_neig)] = 1
    sub_A    = torch.matmul(torch.matmul(operator.t(),A),operator)

    ind    = torch.where(sub_A!=0)
    inds   = torch.cat([ind[0],ind[1]]).reshape(2,len(ind[0]))
    values = torch.ones(len(ind[0]))
    sub_E  = torch.sparse_coo_tensor(inds, values, torch.Size([n_neig, n_neig])).to(A.device)

    return sub_E



def update_A(x_s,neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n,n)
    A = torch.zeros(n*n)

    for i in range(n):
        for j in range(i,n):
            K[i,j] = torch.norm(x_s[i]-x_s[j])
            K[j,i] = K[i,j]
    
    edge  = int(n+torch.round(torch.tensor(neig*n/2)))      
    
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)               
    _, indices = torch.sort(Simil) 
    A[indices[0:edge]] = 1              
    A = A.reshape(n,n).to(x_s.device)

    return A



def update_E(x_s,neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n,n)
    A = torch.zeros(n*n)

    for i in range(n):
        for j in range(i,n):
            K[i,j] = torch.norm(x_s[i]-x_s[j])
            K[j,i] = K[i,j]
    
    edge  = int(n+torch.round(torch.tensor(neig*n/2)))       
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)               
    _, indices = torch.sort(Simil) 
    A[indices[0:edge]] = 1              
    A = A.reshape(n,n)
    ind = torch.where(A==1)

    ind = torch.cat([ind[0],ind[1]]).reshape(2,edge)
    values = torch.ones(edge)
    E = torch.sparse_coo_tensor(ind, values, torch.Size([n,n])).to(x_s.device)

    return E
    
class Aggr(MessagePassing):
    """
    Undirected nodes features aggregation ['add', 'mean']
    """
    def __init__(self, aggr='add'):
        super(Aggr, self).__init__(aggr=aggr)

    def forward(self, x, edge_index):
        """
        inputs:
            x: [N, dim]
            edge_index: [2, edge_num]
        outputs:
            the aggregated node features
            out: [N, dim]
        """
        edge_index = torch.cat([edge_index, edge_index.flip(dims=[0])], dim = 1)
        edge_index = torch.unique(edge_index, dim = 1)
        return self.propagate(edge_index, x=x) + x
    
def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

            
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros([len(g.node_tags), len(tagset)])
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def dense_adj_to_sparse(A):
    """
    Transform dense adjacency matrix to sparse adjacency matrix
    """
    n        = A.shape[0]
    ind    = torch.where(A!=0)
    inds   = torch.cat([ind[0],ind[1]]).reshape(2,len(ind[0]))
    values = torch.ones(len(ind[0])).to(A.device)
    sparse_A  = torch.sparse_coo_tensor(inds, values, torch.Size([n, n])).to(A.device)
    return sparse_A