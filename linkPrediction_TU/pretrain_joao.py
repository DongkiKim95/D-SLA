import argparse

from tu_dataset import TUDatasetExt

from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_geometric.utils as tg_utils

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

class Augmentation():
    def __init__(self, aug_ratio, aug_mode, num_node_type):
        self.num_node_type = num_node_type

        self.augmentations = [self.node_drop, self.subgraph, self.edge_pert, self.attr_mask, lambda x,y:x]

        self.aug_mode = aug_mode
        self.aug_ratio = aug_ratio
        self.set_aug_prob(np.ones(25)/25)

    def __call__(self, data):
        data1 = data.clone()
        data2 = data.clone()

        n_aug = np.random.choice(25, 1, p=self.aug_prob)[0]
        n_aug1, n_aug2 = n_aug//5, n_aug%5
        data1 = self.augmentations[n_aug1](data1, self.aug_ratio)
        data2 = self.augmentations[n_aug2](data2, self.aug_ratio)
        
        return data, data1, data2

    def set_aug_mode(self, aug_mode='none'):
        self.aug_mode = aug_mode

    def set_aug_ratio(self, aug_ratio=0.2):
        self.aug_ratio = aug_ratio
    
    def set_aug_prob(self, prob):
        if prob.ndim == 2:
            prob = prob.reshape(-1)
        self.aug_prob = prob

    
    def node_drop(self, data, aug_ratio):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num  * aug_ratio)

        idx_perm = np.random.permutation(node_num)
        idx_nondrop = idx_perm[drop_num:].tolist()
        idx_nondrop.sort()

        edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

        data.x = data.x[idx_nondrop]
        data.edge_index = edge_index
        data.__num_nodes__, _ = data.x.shape
        return data


    def subgraph(self, data, aug_ratio):
        G = tg_utils.to_networkx(data)

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1-aug_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

        data.x = data.x[idx_nondrop]
        data.edge_index = edge_index
        data.__num_nodes__, _ = data.x.shape
        return data


    def edge_pert(self, data, aug_ratio):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * aug_ratio)

        edge_index = data.edge_index[:, np.random.choice(edge_num, (edge_num - pert_num), replace=False)]

        idx_add = np.random.choice(node_num, (2, pert_num))
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_add[0], idx_add[1]] = 1
        adj[np.arange(node_num), np.arange(node_num)] = 0
        edge_index = adj.nonzero(as_tuple=False).t()

        data.edge_index = edge_index
        return data


    def attr_mask(self, data, aug_ratio):
        node_num, _ = data.x.size()
        mask_num = int(node_num * aug_ratio)
        _x = data.x.clone()

        token = self.num_node_type
        idx_mask = np.random.choice(node_num, mask_num, replace=False)

        _x[idx_mask] = token
        data.x = _x
        return data

class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool

        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss



def train(args, loader, model, optimizer, device, gamma_joao):
    model.train()

    train_loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    # for step, batch in enumerate(loader, desc="Iteration"):
        # _, batch1, batch2 = batch
        _, batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    # joao
    aug_prob = loader.dataset.transform.aug_prob
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.transform.set_aug_prob(_aug_prob)

        count, count_stop = 0, len(loader.dataset)//(loader.batch_size*10)+1 # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            # for step, batch in enumerate(loader, desc="Iteration"):
                # _, batch1, batch2 = batch
                _, batch1, batch2 = batch
    
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.batch)
                x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.batch)
                loss = model.loss_cl(x1, x2)

                loss_aug[n] += loss.item()
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= count

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1/25))
    mu_min, mu_max = b.min()-1/25, b.max()-1/25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b-mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b-mu, 0)
    aug_prob /= aug_prob.sum()

    return train_loss_accum/(step+1), aug_prob


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
    parser.add_argument('--pretrain_data_ratio', type=float, default=0.5)
    parser.add_argument('--output_model_dir', type = str, default = 'ckpts', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')

    parser.add_argument('--aug_mode', type=str, default = 'sample') 
    parser.add_argument('--aug_ratio', type=float, default = 0.2)

    parser.add_argument('--gamma', type=float, default = 0.1)
    args = parser.parse_args()

    # set up output directory
    args.output_model_dir = args.output_model_dir + \
        '/{}/joao/'.format(args.dataset)
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

    if os.path.isfile('./indices/{}/pretrain_indices.pt'.format(args.dataset, args.seed)):
        pretrain_indices = torch.load('./indices/{}/pretrain_indices.pt'.format(args.dataset))
        dataset = dataset[pretrain_indices]
    else:
        pretrain_indices = torch.tensor( np.random.choice(len(dataset), int(len(dataset) * args.pretrain_data_ratio), replace=False), dtype=torch.long)
        
        dataset = dataset[pretrain_indices]
        os.makedirs('./indices/{}/'.format(args.dataset))
        torch.save(pretrain_indices, './indices/{}/pretrain_indices.pt'.format(args.dataset))

    dataset.transform = Augmentation(args.aug_ratio, args.aug_mode, num_node_type)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    model.to(device)


    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    aug_prob = np.ones(25) / 25

    for epoch in range(1, args.epochs+1):
        dataset.transform.set_aug_prob(aug_prob)
        pretrain_loss, aug_prob = train(args, loader, model, optimizer, device, args.gamma)

        print(epoch, pretrain_loss, aug_prob)
        if not args.output_model_dir == "" and epoch % 20 == 0:
            torch.save(model.gnn.state_dict(), args.output_model_dir + "/{}.pth".format(epoch))


if __name__ == "__main__":
    main()

