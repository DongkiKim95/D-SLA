import argparse

from tu_dataset import TUDatasetExt
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import os 

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MaskNode:
    def __init__(self, num_node_type, mask_rate):
        self.num_node_type = num_node_type
        self.mask_rate = mask_rate

    def __call__(self, data):
        num_nodes = data.x.size(0)
        mask_num = int(num_nodes * self.mask_rate) + 1
        mask_idx = torch.tensor( np.random.choice(num_nodes, mask_num, replace=False), dtype=torch.long)

        # masking token
        data.mask_node_label = data.x[mask_idx].flatten().clone()
        
        data.x[mask_idx] = self.num_node_type
        
        data.masked_node_index = mask_idx
        
        return data

criterion = nn.CrossEntropyLoss()

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)


def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_node = model_list
    optimizer_model, optimizer_linear_pred_node = optimizer_list

    model.train()
    linear_pred_node.train()

    loss_accum = 0
    whole_num = 0
    mask_correct = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_rep = model(batch.x, batch.edge_index)

        ## loss for nodes
        pred_node = linear_pred_node(node_rep[batch.masked_node_index])
        loss = criterion(pred_node.double(), batch.mask_node_label)
        

        mask_correct += (pred_node.argmax(dim=-1) == batch.mask_node_label).sum()
        whole_num += pred_node.size(0)
        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        # acc_node_accum += acc_node

        optimizer_model.zero_grad()
        optimizer_linear_pred_node.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_node.step()

        loss_accum += float(loss.cpu().item())

    return loss_accum/step, (mask_correct/whole_num).cpu().item()

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
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'COLLAB', help='root directory of dataset for pretraining')
    parser.add_argument('--pretrain_data_ratio', type=float, default=0.5)
    parser.add_argument('--output_model_dir', type=str, default = 'ckpts', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 2, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("num layer: %d mask rate: %f" %(args.num_layer, args.mask_rate))

    args.output_model_dir = args.output_model_dir + \
        '/{}/masking/'.format(args.dataset)
    print(args.output_model_dir)
    os.makedirs(args.output_model_dir)

    #set up dataset and transform function.

    dataset = TUDatasetExt('./datasets/'+args.dataset, args.dataset)
    num_node_type = dataset.num_node_type()
    if os.path.isfile('./indices/{}/pretrain_indices.pt'.format(args.dataset)):
        pretrain_indices = torch.load('./indices/{}/pretrain_indices.pt'.format(args.dataset))
        dataset = dataset[pretrain_indices]
    else:
        pretrain_indices = torch.tensor( np.random.choice(len(dataset), int(len(dataset) * args.pretrain_data_ratio), replace=False), dtype=torch.long)
        
        dataset = dataset[pretrain_indices]
        os.makedirs('./indices/{}/'.format(args.dataset))
        torch.save(pretrain_indices, './indices/{}/pretrain_indices.pt'.format(args.dataset))
    dataset.transform = MaskNode(num_node_type=num_node_type, mask_rate=args.mask_rate)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    linear_pred_node = torch.nn.Linear(args.emb_dim, int(num_node_type)).to(device)

    model_list = [model, linear_pred_node]

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_node = optim.Adam(linear_pred_node.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_node]

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc_atom = train(args, model_list, loader, optimizer_list, device)
        print(train_loss, train_acc_atom)

        if not args.output_model_dir == "" and epoch % 20 == 0:
            torch.save(model.state_dict(), args.output_model_dir + "/{}.pth".format(epoch))

if __name__ == "__main__":
    main()
