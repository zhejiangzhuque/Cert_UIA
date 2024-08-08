import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj
import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from data.feats import compute_x
from sklearn import manifold
import math
from sklearn.model_selection import StratifiedKFold

from data import NormalizedFeat, PadFeat


def train_eval_test_split(args, dataset, train_indices=[], val_indices=[], test_indices=[], train_graphs=[], val_graphs=[], seed=2042, **kwargs):
    data_size = len(dataset)
    print('data size:', data_size)
    indices = list(range(data_size))
    if len(train_indices) == 0:
        train_size = int(data_size*args.data['train_rate'])
        eval_size = (data_size-train_size)//2
        test_size = data_size - train_size - eval_size
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+eval_size]
        test_indices = indices[train_size+eval_size:]
    else:
        assert len(val_indices)>0 and len(test_indices)>0, "val and test indices must be non-empty!"
        
    train_set = dataset[train_indices]
    eval_set = dataset[val_indices]
    test_set = dataset[test_indices]

    train_x = []
    for g in train_set:
        gx = g.x
        if kwargs.get('dense', False):
            gx = F.pad(gx, (0,0,0,kwargs['max_node']-gx.shape[0]))
        train_x.append(gx)
    train_x = torch.cat(train_x, dim=0)
    mean = torch.mean(train_x, dim=0)
    std = torch.std(train_x, dim=0)

    transform = T.Compose([NormalizedFeat(mean, std),PadFeat(args.data['pad_feat'])])
    
    def _transform(data_set, offset=0):
        output = []
        for i, data in enumerate(data_set):
            data = transform(data)
            data.id = offset + i
            output.append(data)
        output = Batch.from_data_list(output)
        return output

    if args.data['dataset'] in ["IMDB-BINARY"]:
        print("mean:", mean, "std:", std)
        train_set = _transform(train_set)
        eval_set = _transform(eval_set, len(train_set))
        test_set = _transform(test_set, len(train_set)+len(eval_set))
    
    print('train size:', len(train_set), 'eval_size:', len(eval_set), 'test_size:', len(test_set))

    batch_size = args.training['batch_size']
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader, test_loader


def k_fold(dataset,args):
    train_indices, val_indices, test_indices = load_fold_idx()
    if len(train_indices) > 0:
        print("find kflod indices, load directly!")
        return train_indices, val_indices, test_indices

    kf = StratifiedKFold(args.n_fold, shuffle=True, random_state=42)

    for _, idx in kf.split(torch.zeros(len(dataset)), dataset.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(args.n_fold)]

    for i in range(args.n_fold):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    save_fold_idx(train_indices, val_indices, test_indices)
    
    return train_indices, val_indices, test_indices

def save_fold_idx(train_inds, val_inds, test_inds, args):
    fold_path = f"{args.data_path}/{args.dataset}/{args.n_fold}fold_idx"
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    for fold in range(args.n_fold):
        np.savetxt(f'{fold_path}/train_idx-{fold+1}.txt', train_inds[fold], fmt='%d')
        np.savetxt(f'{fold_path}/val_idx-{fold+1}.txt', val_inds[fold], fmt='%d')
        np.savetxt(f'{fold_path}/test_idx-{fold+1}.txt', test_inds[fold], fmt='%d')

def load_fold_idx(args):
    train_inds, val_inds, test_inds = [], [], []
    fold_path = f"{args.data_path}/{args.dataset}/{args.n_fold}fold_idx"
    if os.path.exists(fold_path):
        for fold in range(args.n_fold):
            train_inds.append(np.loadtxt(f'{fold_path}/train_idx-{fold+1}.txt', dtype=int))
            val_inds.append(np.loadtxt(f'{fold_path}/val_idx-{fold+1}.txt', dtype=int))
            test_inds.append(np.loadtxt(f'{fold_path}/test_idx-{fold+1}.txt', dtype=int))
    return train_inds, val_inds, test_inds


def inject_nodes(dataset, n_nodes=1, alpha=0.5):
    injected_graphs = []
    np.random.seed(2345)
    torch.manual_seed(2345)
    for i in range(len(dataset)):
        data = dataset[i]
        x = data.x
        adj = to_dense_adj(data.edge_index, max_num_nodes=x.shape[0])[0]
        adj = adj + torch.eye(adj.shape[0])
        x_agg = torch.matmul(adj, x)/adj.sum(dim=1,keepdim=True)
        N = len(x)
        inds = list(range(N))
        n_nodes = min(N, n_nodes)
        injected_indices = np.random.choice(inds, n_nodes)
        x_injected = (1-alpha)*x_agg[injected_indices] + alpha*torch.rand(len(injected_indices),x.shape[1])
        x = torch.cat([x_injected,x],dim=0)
        edge_index_injected = []
        for i in range(len(injected_indices)):
            j = injected_indices[i]+n_nodes
            edge_index_injected += [[i,j],[j,i]]
        edge_index_injected = torch.LongTensor(edge_index_injected).t().contiguous()
        edge_index = torch.cat([data.edge_index+n_nodes, edge_index_injected],dim=1)
        injected_graphs.append(Data(x=x,edge_index=edge_index,y=data.y))
    dataset = Batch.from_data_list(injected_graphs)
    return dataset


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, model=None, device=None):
    if not os.path.exists(path):
        raise ValueError(f"Incorrect model path: {path}")
    if device is not None:
        ckpt = torch.load(path, map_location=device)
    else:
        ckpt = torch.load(path)
    if model is not None:
        model.load_state_dict(ckpt)
    return ckpt

def calculate_margin(output, target):
    y = output.argmax(1)
    label_mask = y.eq(target)
    output = output[label_mask]
    margin = torch.abs(output[:,0]-output[:,1])

    return margin

def calculate_sim(data, sim_list):
    # adj_node similarity
    graph_sim = 0.
    for j in range(len(Batch.to_data_list(data))):
        feature_matrix = Batch.to_data_list(data)[j].x
        edge_matrix = Batch.to_data_list(data)[j].edge_index

        mean_sim = 0.
        count = 0
        for t in range(feature_matrix.shape[0]):
            neighbors = edge_matrix[1][edge_matrix[0]==t]
            if neighbors.shape[0]==0:
                continue
            else:
                count += 1
                sim = torch.mm(feature_matrix[t].unsqueeze(0), torch.transpose(feature_matrix[neighbors], 0, 1))
                sim = (sim.sum()/sim.shape[1]).item()
                mean_sim += sim
        mean_sim /= count 
        graph_sim += mean_sim
    
    sim_list.append((graph_sim/len(Batch.to_data_list(data))))
    return sim_list


def show_graph(advg,inj_nums,index=0,direct=True):
    if inj_nums == 0:
        return
    
    def _draw(edges_, inj_nodes):
        G = nx.Graph()
        inj_nodes = set(inj_nodes)
        for x,y in edges_:
            if x in inj_nodes or y in inj_nodes:
                G.add_edge(x,y,style='dashed')
            else:
                G.add_edge(x,y,style='solid')

        orig_nodes = []
        orig_nodes = set(G.nodes())-inj_nodes
        pos = nx.spring_layout(G)
        edge_styles = [G[u][v]['style'] for u,v in G.edges]
        nx.draw(G, pos, node_size=160, font_size=10, style=edge_styles, with_labels=True)
        nx.draw_networkx_nodes(G, pos, nodelist=orig_nodes,node_color='b')
        nx.draw_networkx_nodes(G, pos, nodelist=inj_nodes,node_color='r')
        
    edges = advg.edge_index.t().data.cpu().numpy()
    inject_nodes = np.arange(inj_nums)
    _draw(edges, inject_nodes)
    
    plt.show()

def tsne_vis(data, codes, name='tsne'):
    def _normalize(x):
        mean = np.mean(x, 0, keepdims=True)
        std = np.std(x, 0, keepdims=True)
        x = (x-mean)/(std+1e-10)
        return x
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=242)
    x = tsne.fit_transform(data)
    x = _normalize(x)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    
    unicodes = np.unique(codes)
    mask = np.zeros(len(unicodes))
    colors = plt.cm.rainbow(np.linspace(0,1,len(unicodes)))

    for j in range(len(x)):
        if mask[codes[j]] == 0:
            mask[codes[j]] = 1
            plt.scatter(x[j,0],x[j,1],color=colors[codes[j]],s=60,label=str(codes[j]))
        else:
            plt.scatter(x[j,0],x[j,1],s=60,color=colors[codes[j]])

    plt.legend()
    plt.savefig(name+'.png')
    plt.show()


def norm_adj(adj):
    N = adj.shape[0]
    adj = adj + torch.eye(N).to(adj)
    deg = adj.sum(dim=1)
    deg = deg**(-0.5)
    deg[torch.isinf(deg)] = 0
    D = torch.diag(deg)
    adj = D @ adj @ D
    return adj


dense_caches = {}
def convert_sparse_to_dense(data_batch, max_node):
    feats = []
    adjs_norm = []
    adjs_ori = []
    for data in Batch.to_data_list(data_batch):
        cache_idx = data.id
        if cache_idx in dense_caches:
            feats.append(dense_caches[cache_idx][0])
            adjs_norm.append(dense_caches[cache_idx][1])
            adjs_ori.append(dense_caches[cache_idx][2])
            continue
        x_padded = F.pad(data.x,(0,0,0,max_node-data.x.shape[0]))
        adj = to_dense_adj(data.edge_index, max_num_nodes=max_node)[0]
        adj_norm = norm_adj(adj)
        feats.append(x_padded)
        adjs_norm.append(adj_norm)
        adjs_ori.append(adj)
        dense_caches[cache_idx] = (feats[-1], adjs_norm[-1], adjs_ori[-1])
    feats = torch.stack(feats)
    adjs_norm = torch.stack(adjs_norm)
    adjs_ori = torch.stack(adjs_ori)

    return feats, adjs_norm, adjs_ori


def select_topk_rate(model, val_loader, device, sample_times=10, dense=False, **kwargs):
    model.eval()
    
    best_margin = np.inf
    best_k = 1.0
    with torch.no_grad():
        for j in range(sample_times):
            k = np.random.rand() * 0.9 + 0.05
            margin_list = []
            for i, data in enumerate(val_loader):
                data = data.to(device)
                if dense:
                    data_ = convert_sparse_to_dense(data,kwargs['max_node'])
                else:
                    data_ = data
                output, output_adv, _, _, adv_g,_,_ = model(data_, k)

                margin = calculate_margin(output_adv, data.y)
                margin_list.append(margin.cpu())

            margin_list = torch.cat(margin_list)
            margin = sum(margin_list)/len(margin_list)
            if margin < best_margin:
                best_margin = margin
                best_k = k
    return best_k, best_margin
    
    
def get_lr_scheduler(args, scheduler_name, optimizer, **kwargs):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch, eta_min=1e-4, last_epoch=-1, verbose=kwargs['verbose'])
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, math.ceil(args.n_epoch*0.5), gamma=0.1, last_epoch=-1, verbose=kwargs['verbose'])
    if scheduler_name == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, kwargs['milestones'], gamma=0.1, last_epoch=-1, verbose=kwargs['verbose'])

    return None