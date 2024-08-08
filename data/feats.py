import torch
import numpy as np
import networkx as nx
from numpy import linalg as LA
from sklearn.preprocessing import OneHotEncoder
import argparse


def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])
    
    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]

    for i in g.nodes():
        nodes = g[i].keys()

        nbrs_deg = [g.nodes[j][key] for j in nodes]

        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]

    return x

def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)


def compute_x(a1, node_features, **kwargs):
    # construct node features X
    if node_features == 'identity':
        x = torch.cat([torch.diag(torch.ones(a1.shape[1]))] * a1.shape[0]).reshape([a1.shape[0], a1.shape[1], -1])
        x1 = x.clone()

    elif node_features == 'degree':
        a1b = (a1 != 0).float()
        x1 = a1b.sum(dim=2, keepdim=True)

    elif node_features == 'degree_bin':
        a1b = (a1 != 0).float()
        x1 = binning(a1b.sum(dim=2))

    elif node_features == 'adj': # edge profile
        x1 = a1.float()

    elif node_features == 'LDP': # degree profile
        a1b = (a1 != 0).float()
        x1 = []
        n_graphs: int = a1.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))
        x1 = np.array(x1)

    elif node_features == 'eigen':
        _, x1 = LA.eig(a1.numpy())

    elif node_features == 'eigen_topk':
        a = a1.numpy()
        lambdas, eigvs = LA.eig(a)
        lambdas = np.real(lambdas)
        eigvs = np.real(eigvs)
        
        indices = np.argsort(-lambdas, axis=1)  # descending
        eigvsT = eigvs.transpose(0,2,1)
        eigvs_topk = []
        topk = min(kwargs['topk'], lambdas.shape[1])
        for i in range(len(indices)):
            eigvs_topk.append(np.expand_dims(eigvsT[i,indices[i,:topk]],0))

        eigvs_topk = np.concatenate(eigvs_topk, axis=0)  #(B,K,n)
        x1 = np.matmul(eigvs_topk, a).transpose(0,2,1)  # (B,n,K)
        
    x1 = torch.Tensor(x1).float()
    return x1

if __name__ == '__main__':
    a = torch.rand(3,5,5)
    a1 = a.numpy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_features", "-f", default='identity')
    parser.add_argument("--dataset_name")
    parser.add_argument("--modality")
    args = parser.parse_args()
    out = compute_x(a, args)