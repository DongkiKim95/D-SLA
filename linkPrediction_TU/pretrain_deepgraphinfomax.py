import argparse

from tu_dataset import TUDatasetExt

from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

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

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class Infomax(nn.Module):
    def __init__(self, gnn, discriminator):
        super(Infomax, self).__init__()
        self.gnn = gnn
        self.discriminator = discriminator
        self.loss = nn.BCEWithLogitsLoss()
        self.pool = global_mean_pool


def train(args, model, device, loader, optimizer):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_emb = model.gnn(batch.x, batch.edge_index)
        summary_emb = torch.sigmoid(model.pool(node_emb, batch.batch)) # (B, dim)

        positive_expanded_summary_emb = summary_emb[batch.batch] # (node, dim)

        shifted_summary_emb = summary_emb[cycle_index(len(summary_emb), 1)] # (B, dim) shifted
        negative_expanded_summary_emb = shifted_summary_emb[batch.batch] # making negative by shifting

        positive_score = model.discriminator(node_emb, positive_expanded_summary_emb)
        negative_score = model.discriminator(node_emb, negative_expanded_summary_emb)      

        optimizer.zero_grad()
        loss = model.loss(positive_score, torch.ones_like(positive_score)) + model.loss(negative_score, torch.zeros_like(negative_score))
        loss.backward()

        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/step, train_loss_accum/step


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
    parser.add_argument('--num_workers', type=int, default = 2, help='number of workers for dataset loading')
    args = parser.parse_args()

    # set up output directory
    args.output_model_dir = args.output_model_dir + \
        '/{}/dgi'.format(args.dataset)
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


    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    discriminator = Discriminator(args.emb_dim)

    model = Infomax(gnn, discriminator)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, loader, optimizer)

        print(train_acc)
        print(train_loss)

        if not args.output_model_dir == "" and epoch % 20 == 0:
            torch.save(gnn.state_dict(), args.output_model_dir + "/{}.pth".format(epoch))

if __name__ == "__main__":
    main()
