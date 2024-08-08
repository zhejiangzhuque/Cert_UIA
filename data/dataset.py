import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj
import numpy as np

from .feats import compute_x
from .dfs import dfs


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class FeatTransformAdj(object):
    def __init__(self, max_node, feature_type):
        self.max_node = max_node
        self.feture_type = feature_type

    def __call__(self, data):
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)
        x = compute_x(adj, self.feture_type)[0]
        data.x = x

        return data
    
class FeatTransformDeg(object):
    def __init__(self, mean, std, max_degree):
        self.mean = mean
        self.std = std
        self.max_degree = max_degree

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.max_degree+1)
        data.x = torch.cat([data.x, deg.view(-1, self.max_degree+1)],dim=-1)
        
        return data

class NormalizedFeat(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.x = (data.x - self.mean) / self.std
        return data
    
class PadFeat(object):
    def __init__(self, pad_num):
        self.pad_num = pad_num

    def __call__(self, data):
        data.x = torch.cat([data.x, torch.zeros(data.x.shape[0], self.pad_num)], dim=-1)

        return data


def load_dataset(dataset_name, path):
    graphs = TUDataset(path, dataset_name)
                           
    if graphs.data.x is None:
        max_degree = 0
        degs = []
        for data in graphs:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 2000:
            graphs.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            graphs.transform = NormalizedDegree(mean, std)
    
    max_node = 0
    labels = []
    for i in range(len(graphs)):
        data = graphs[i]
        labels.append(int(data.y[0]))
        max_node = max(max_node, data.x.shape[0])
    print("max node:", max_node)
    labels = np.array(labels)
    for l in np.unique(labels):
        print("label %d:"%l, np.sum(labels==l))

    degs = []
    max_degree = 0
    for data in graphs:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())
    deg = torch.cat(degs,dim=0).to(torch.float)
    print("max degree:", max_degree)
    mean, std = deg.mean().item(), deg.std().item()

    if dataset_name in ["IMDB-BINARY"]:
        graphs.transform = FeatTransformAdj(max_node, "LDP")

    connected = []
    for i, g in enumerate(graphs):
        if dfs(g)[0]:
            connected.append(i)
    print(f'connected: {len(connected)}/{len(graphs)}')
    
    graphs = graphs[connected]

    return graphs, max_node
