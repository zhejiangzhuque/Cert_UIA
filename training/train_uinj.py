import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from sklearn.metrics import accuracy_score
import random
import networkx as nx
import matplotlib.pyplot as plt

import os
import os.path as osp

from models import UInj, BaseModel
from data import load_dataset
from utils import save_model, load_model, train_eval_test_split, calculate_margin

from args import get_args
args = get_args("train_uinj.yaml")
f_args = get_args('pretrain_F.yaml')
base_args = get_args('train_gcn.yaml')

args.save_dir = osp.join(args.output_dir, args.data['dataset'], "checkpoints")

if not osp.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = torch.device(args.device)

def margin_loss(output, target, epsilon=0.0001):
    n_correct = len(output)
    y = output[range(n_correct), target]
    mask = torch.ones_like(output)
    mask[range(n_correct), target] = 0
    mask = mask.bool()
    y_sec = output[mask].view(n_correct, -1).max(dim=1)[0]

    loss = torch.clamp(y-y_sec-epsilon, min=0).sum() / y.shape[0]
    return loss

def cls_loss(output, target):
    loss = F.cross_entropy(output, target)
    return loss 

def loss_func(output, target, deltaE):
    mask = (output.argmax(1)==target)
    l_mag = margin_loss(output[mask], target[mask])
    l_ce = cls_loss(output, target)
    l_E = F.l1_loss(deltaE, torch.zeros_like(deltaE))
    loss = 2*l_ce + 1*l_mag + 1*l_E
    return loss


def train(model, loaders):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    best_acc = -1
    accs = []

    for epoch in range(args.training['n_epoch']):
        model.train()
        for i, data in enumerate(loaders[0]):
            data = data.to(device)
            output, deltaE, _ = model(data)
            target = data.y
            
            loss = loss_func(output, target, deltaE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f"[Train] Epoch {epoch+1}, step {i+1}, loss: {loss:.4f}")
                losses.append(loss.item())
        acc, margin = evaluate(model, loaders[1])
        print(f'[Evaluate] Epoch {epoch+1}, acc:{acc:.4f}')
        if acc > best_acc+1e-12:
            best_acc = acc
            save_model(model, osp.join(args.save_dir, "best.pt"))
        accs.append(acc)
    save_model(model, osp.join(args.save_dir, "model.pt"))
    return losses, accs


def evaluate(model, data_loader, load=False):
    if load:
        print("load", args.test['load'])
        load_model(osp.join(args.save_dir, args.test['load']), model)
        
    model.eval()
    pred = []
    gt = []
    margin_list = []
    inj_node = 0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            output, deltaE, p1 = model(data)
            
            pred.append(output.argmax(1))
            gt.append(data.y)
            margin = calculate_margin(output, data.y)
            margin_list.append(margin.cpu())

            node_list = []

            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                node_list.append((deltaE[j][:N] > 0.5).sum().item())
                inj_node += (deltaE[j][:N] > 0.5).sum().item()   

    pred = torch.cat(pred)
    gt = torch.cat(gt)
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()

    acc = accuracy_score(gt, pred)

    margins = torch.cat(margin_list)
    margin = sum(margins)/len(margins)

    # Average inject node nums
    avg_inj_num = inj_node / len(gt)
    
    print(f'[Test] acc: {acc:.4f}, margin: {margin:.4f}, avg_inj: {avg_inj_num:.2f}')
    
    return acc, margin

def plot(losses, accs):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.plot(losses)
    plt.xlabel('steps')
    plt.title('loss')
    plt.subplot(122)
    plt.plot(accs)
    plt.title('acc')
    plt.xlabel('epoch')
    plt.savefig(osp.join(osp.dirname(args.save_dir), 'training.png'),dpi=200,bbox_inches='tight',pad_inches=0.1)
    plt.show()


def run():
    # UINJ model
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    losses, accs = train(model, loaders)
    evaluate(model, loaders[2], load=True)

    plot(losses, accs)


def test():
    # UINJ model
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    evaluate(model, loaders[2], load=True)


def exp():
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    dataset, _ = load_dataset(base_args.data['dataset'], base_args.data['path'])
    loaders = train_eval_test_split(base_args, dataset, seed=base_args.seed)
    base_model = BaseModel(dataset.num_node_features, base_args.model['hidden_size'], dataset.num_classes)
    base_model.to(device)

    load_model(osp.join(args.save_dir, args.test['load']), model)
    load_model(osp.join(osp.join(base_args.output_dir, base_args.data['dataset'], "checkpoints"), base_args.test['load']), base_model)
    
    model.eval()
    base_model.eval()

    pred = []
    gt = []
    pred_new = []
    rho_list = []
    perturb_data = []    

    with torch.no_grad():
        for i, data in enumerate(loaders[2]):
            data = data.to(device)
            output, deltaE, p1 = model(data)

            pred.append(output.argmax(1))
            gt.append(data.y)

            deltaE_bin = torch.where(deltaE > 0.5, torch.ones_like(deltaE), torch.zeros_like(deltaE))

            new_data = []
            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                xt = p1[j] * N
                x_new = torch.cat([graph.x, xt.unsqueeze(0)], dim=0)
                
                rho_list.append((deltaE[j][:N] > 0.5).sum().item())
                perturb_data.append(Data(x=x_new, edge_index=graph.edge_index, y=graph.y))

                edge_index = graph.edge_index
                added_edges = []
                row_vector = deltaE_bin[j]
                zero_indices = torch.nonzero(row_vector[:N] == 0).squeeze(1)
                if zero_indices.size(0)==0:
                    new_data.append(Data(x=x_new, edge_index=edge_index))
                    continue
                rand_index = torch.randint(high=zero_indices.size(0), size=(1,)).item()
                index = zero_indices[rand_index].item()
                added_edges.append([N, index])
                added_edges.append([index, N])
                added_edges = torch.LongTensor(added_edges).t().contiguous()

                edge_index = torch.cat([edge_index, added_edges.to(edge_index.device)], dim=1)

                new_data.append(Data(x=x_new, edge_index=edge_index))
            
            new_graph_batch = Batch.from_data_list(new_data).to(device)
            output_new, _ = base_model(new_graph_batch)
            pred_new.append(output_new.argmax(1))
    
    # save perturb data
    file_path = "./output/perturb_data/"+args.data['dataset']+".pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(perturb_data, file_path)
    
    pred = torch.cat(pred)
    pred_new = torch.cat(pred_new)
    gt = torch.cat(gt)
    mask1 = (pred==gt)
    mask2 = (pred_new==pred)
    acc_before = torch.sum(mask1).item()/len(mask1)
    acc_after = torch.sum(mask2).item()/len(mask2)

    print(f'Robust_acc: {acc_before:.4f}, flip_rate: {1-acc_after:.4f}')

    rho_list = torch.tensor(rho_list)
    rho_list = rho_list[mask1]
    acc_0 = len(rho_list) / len(pred)
    sort_rho,_ = torch.sort(rho_list)

    print(f'rho=0, acc: {acc_0:.4f}')
    print(f'The perturbation radius list: {sort_rho}')