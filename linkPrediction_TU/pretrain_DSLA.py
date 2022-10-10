import argparse

from tu_dataset import TUDatasetExt

from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_geometric.utils as tg_utils
from torch_geometric.data import Batch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class DSLAPerturbation():
    def __init__(self, num_node_type, args):
        self.args = args
        self.num_node_type = num_node_type

        self.edge_pert_strength = args.edge_pert_strength
        self.mask_strength = args.mask_strength

        self.k = args.k
        self.num_candidates = args.num_candidates
        self.edit_learn = args.edit_learn

    def __call__(self, data):
        candidates = [data.clone() for _ in range(self.num_candidates)]
        
        # subgraph sampling
        node_num, _ = data.x.size()
        
        edge_num = data.edge_index.size(1)
        pert_num = int(edge_num * self.edge_pert_strength)
        if pert_num == 0:
            pert_num = 1

        for i in range(self.num_candidates):
            # assign default edit distance
            candidates[i].edit_dist = 0
    
            candidates[i] = self.attr_mask(candidates[i])

            if i > 0:
                if self.edit_learn:
                    curr_pert_num = pert_num * i
                else: curr_pert_num = pert_num
                
                candidates[i] = self.edge_pert(candidates[i], curr_pert_num)
        
        return candidates



    def edge_pert(self, data, pert_num):
        node_num = data.x.size(0)
        edge_num = data.edge_index.size(1)

        possible_edge_num = node_num * (node_num - 1) - edge_num
        pert_num = min(possible_edge_num, pert_num)
        
        idx_chosen = np.random.choice(edge_num, (edge_num - pert_num), replace=False)

        edge_index = data.edge_index[:, idx_chosen]

        node_num = data.x.size(0)
        adj = torch.ones((node_num, node_num))
        adj[range(node_num), range(node_num)] = 0

        adj[data.edge_index[0], data.edge_index[1]] = 0
        edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        
        idx_add = np.random.choice(edge_index_nonexist.shape[1], pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]

        edge_index = torch.cat([edge_index, edge_index_add], dim=-1)
        data.edge_index = edge_index
        data.edit_dist = pert_num * 2

        return data


    def subgraph_edge_pert(self, data, pert_num, sub_edge_num, sub_edge_index, subset, edge_mask):
        # print(sub_edge_num, pert_num)
        idx_chosen = np.random.choice(sub_edge_num, (sub_edge_num - pert_num), replace=False)

        # Remove edges
        sub_edge_index = sub_edge_index[:, idx_chosen]
        edge_index = torch.cat([data.edge_index[:,~edge_mask], sub_edge_index], dim=-1)

        # Add edges in sampled subgraph
        node_num = data.x.size(0)
        adj = torch.ones((node_num, node_num))
        adj[range(node_num), range(node_num)] = 0
        
        not_subset = np.setdiff1d(np.arange(node_num), subset)
        adj[not_subset,:] = 0
        adj[:, not_subset] = 0
        adj[sub_edge_index[0], sub_edge_index[1]] = 0
        edge_index_nonexist = adj.nonzero(as_tuple=False).t()

        idx_add = np.random.choice(edge_index_nonexist.shape[1], pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]

        edge_index = torch.cat([edge_index, edge_index_add], dim=-1)
        data.edge_index = edge_index
        data.edit_dist = pert_num * 2

        return data

    def attr_mask(self, data):
        
        node_num, _ = data.x.size()
        mask_num = int(node_num * self.mask_strength)

        idx_mask = np.random.choice(node_num, mask_num, replace=False)

        token = self.num_node_type
        idx_mask = torch.tensor(idx_mask, dtype=torch.long)
        
        _x = data.x.clone()
        _x[idx_mask] = token
        data.x = _x

        return data

    
class DSLA(nn.Module):
    def __init__(self, gnn, args):
        super(DSLA, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool

        self.score_func = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True),
                                        nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True),
                                        nn.Linear(args.emb_dim, 1))

        if args.edit_learn:
            self.edit_idx1 = []
            self.edit_idx2 = []
            for i in range(args.num_candidates-2):
                self.edit_idx1.extend([i]*(args.num_candidates-i-2))
                for j in range(i+1, args.num_candidates-1):
                    self.edit_idx2.append(j)        

            self.edit_coeff = args.edit_coeff


    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        scores = self.score_func(x)
        return x, scores

    def GD_loss(self, scores, device):
        answer = torch.zeros((scores.size(0),), dtype=torch.long).to(device)

        pred = scores.argmax(dim=-1)
        correct = (pred == answer).sum().cpu().item()
        total = scores.size(0)

        GD_loss = F.cross_entropy(scores, answer)

        return GD_loss, correct, total

    def edit_loss(self, graph_rep, edit_dist):
        # graph_emb : (B, num_candidates, dim)
        rep_diff = (graph_rep[:,1:] - graph_rep[:,:1]).norm(dim=-1)
        
        edit_dist = edit_dist[:, 1:]
        
        dist_norm = rep_diff / edit_dist
        
        edit_loss = F.mse_loss(dist_norm[:, self.edit_idx1], dist_norm[:, self.edit_idx2])
        
        return self.edit_coeff * edit_loss

def list_collate_fn(data_list):
    num_candidates = len(data_list[0])
    
    data_full = []
    for d in data_list:
        data_full.extend(d)

    batch = Batch.from_data_list(data_full)
    
    return batch


def train(args, loader, model, optimizer, device):
    model.train()
    train_loss_accum = 0
    GD_loss_accum, edit_loss_accum = 0, 0
    GD_total_correct, GD_total_num = 0, 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        optimizer.zero_grad()

        batch = batch.to(device)
        GD_loss, edit_loss = 0, 0

        graph_rep, scores = model(batch.x, batch.edge_index, batch.batch)
        
        scores = scores.view(-1, args.num_candidates)
        graph_rep = graph_rep.view(-1, args.num_candidates, args.emb_dim)

        # Graph discrimination
        GD_loss, GD_correct, GD_total = model.GD_loss(scores, device)
        GD_total_correct += GD_correct
        GD_total_num += GD_total

        GD_loss_accum += float(GD_loss.detach().cpu().item())

        # Graph edit distance-based discrepancy learning
        if args.edit_learn:
            edit_loss = model.edit_loss(graph_rep, batch.edit_dist.view(-1, args.num_candidates))
            edit_loss_accum += float(edit_loss.detach().cpu().item())
        
        loss = GD_loss + edit_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss_accum += float(loss.detach().cpu().item())
    
    return train_loss_accum/(step+1), GD_loss_accum/(step+1), \
           edit_loss_accum/(step+1), GD_total_correct/GD_total_num


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'COLLAB', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_dir', type = str, default = 'ckpts/', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--pretrain_data_ratio', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default = 16, help='number of workers for dataset loading')
    
    parser.add_argument('--mask_strength', type=float, default = 0.2)
    parser.add_argument('--edge_pert_strength', type=float, default = 0.1)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--num_candidates', type=int, default=4)

    # embedding-level accurate discrepancy learning
    parser.add_argument('--edit_learn', action='store_true')
    parser.add_argument('--edit_coeff', type=float, default=0.7)

    args = parser.parse_args()

    # set up output directory
    args.output_model_dir = args.output_model_dir + \
        '/{}/DSLA/'.format(args.dataset)
    print(args.output_model_dir)
    os.makedirs(args.output_model_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #set up dataset
    dataset = TUDatasetExt("./datasets/" + args.dataset, args.dataset)
    num_node_type = dataset.num_node_type()

    if os.path.isfile('./indices/{}/pretrain_indices.pt'.format(args.dataset)):
        pretrain_indices = torch.load('./indices/{}/pretrain_indices.pt'.format(args.dataset))
        dataset = dataset[pretrain_indices]
    else:
        pretrain_indices = torch.tensor( np.random.choice(len(dataset), int(len(dataset) * args.pretrain_data_ratio), replace=False), dtype=torch.long)
        
        dataset = dataset[pretrain_indices]
        os.makedirs('./indices/{}/'.format(args.dataset))
        torch.save(pretrain_indices, './indices/{}/pretrain_indices.pt'.format(args.dataset))

    dataset.transform = DSLAPerturbation(num_node_type, args)

    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                    shuffle=True, collate_fn=list_collate_fn)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = DSLA(gnn, args)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        pretrain_loss, GD_loss, edit_loss, GD_acc = \
                        train(args, loader, model, optimizer, device)

        print("Epoch:{}, Loss:{}, GD Loss:{}, Edit Loss:{}, GD Acc:{}".format(epoch, \
                        pretrain_loss, GD_loss, edit_loss, GD_acc*100))

        if not args.output_model_dir == "" and epoch % 20 == 0:
            torch.save(model.gnn.state_dict(), args.output_model_dir + "/{}.pth".format(epoch))

if __name__ == "__main__":
    main()

