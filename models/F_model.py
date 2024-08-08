import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from models import BaseModel
from utils import load_model, save_model

import os.path as osp


class Sampler(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(Sampler, self).__init__()
        self.mlp_mu = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

        self.mlp_sigma = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        
    def forward(self, x):
        mu = self.mlp_mu(x)
        sigma = self.mlp_sigma(x)

        return torch.randn_like(mu) * torch.exp(sigma/2) + mu, self.loss(mu, sigma)   # VAE: reparameter trick

    def loss(self, mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp())    # negative log-likelihood loss


class FModel(nn.Module):
    def __init__(self, maxN, inp_size, out_size, args, base_args):
        super(FModel, self).__init__()

        self.sampler = Sampler(base_args.model['hidden_size'], args.sampler['hidden_size'], inp_size)

        self.base = BaseModel(inp_size, base_args.model['hidden_size'], out_size)
        load_model(args.load, self.base)
        for p in self.base.parameters():
            p.requires_grad = False

        self.F = nn.Sequential(
            nn.Linear(base_args.model['hidden_size'], args.model['hidden_size']),
            nn.ReLU(),
            nn.Linear(args.model['hidden_size'], maxN),
            nn.Sigmoid()    
        )

        self.args = args
        self.base_args = base_args

    def _sample_edges(self, p, graph):
        N = graph.x.shape[0]
        indices = torch.where(torch.rand(N)>0.5)[0]
        while len(indices) == 0:
            indices = torch.where(torch.rand(N)>0.5)[0]
        edge_index = graph.edge_index
        added_edges = []
        for i in indices:
            added_edges.append([N, i])
            added_edges.append([i, N])
        added_edges = torch.LongTensor(added_edges).t().contiguous()
        edge_index = torch.cat([edge_index, added_edges.to(edge_index.device)], dim=1)
        p = p*N
        x = torch.cat([graph.x, p.unsqueeze(0)], dim=0)
        new_graph = Data(x=x, edge_index=edge_index, y=graph.y)
        deltaE = torch.zeros(N).to(x.device)
        deltaE[indices] = 1.0
        return new_graph, deltaE

    def _loss(self, output, y, mask):
        return torch.nn.functional.binary_cross_entropy(output*mask, y)
    
    def _train(self, p, graph_batch):
        p = p.detach()
        re_graph_list = []
        deltaE_list = []
        for i, graph in enumerate(Batch.to_data_list(graph_batch)):
            for _ in range(self.args.sample_num):
                new_graph, deltaE = self._sample_edges(p[i], graph)
                re_graph_list.append(new_graph)
                deltaE_list.append(deltaE)

        new_graph_batch = Batch.from_data_list(re_graph_list)
        node_number = [len(E) for E in deltaE_list]
        maxN = max(node_number)
        padded_deltaE = []
        edge_mask = []
        for E in deltaE_list:
            padded_deltaE.append(torch.nn.functional.pad(E, (0, maxN-len(E))))
            mask = torch.zeros(maxN)
            mask[:len(E)] = 1.0
            edge_mask.append(mask)
        deltaE = torch.stack(padded_deltaE)
        edge_mask = torch.stack(edge_mask).to(deltaE.device)

        _, gx = self.base(graph_batch)
        _, gx_new = self.base(new_graph_batch)
        gx = gx.unsqueeze(1).repeat(1, self.args.sample_num,1).view(-1, gx.shape[-1])
        deltaH = gx_new - gx

        batch_size = self.args.training['batch_size']
        optimzer = torch.optim.Adam(self.F.parameters(), lr=self.args.training['lr_F'])
        for i in range(1):
            for j in range(0, len(deltaH), batch_size):
                output = self.F(deltaH[j: j+batch_size])
                loss = self._loss(output[:, :maxN], deltaE[j: j+batch_size], edge_mask[j: j+batch_size])
                print(f"F-loss (step {j//batch_size}): {loss:.4f}")
                
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
            
    def forward(self, data):
        _, gx = self.base(data)
        p, p_loss = self.sampler(gx)
        self._train(p, data)

        return p, p_loss
