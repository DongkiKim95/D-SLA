import argparse

from tu_dataset import TUDatasetExt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

#from model import GNN
from model import GNN
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data, DataLoader

import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def ContextExtract(data):
    num_nodes = data.num_nodes

    ego_indices = data.ego_indices

    if len(ego_indices) > 0:
        center_idx = np.random.choice(ego_indices, 1)[0]
    else:
        center_idx = np.random.choice(num_nodes, 1)[0]
    data.center_index = torch.tensor( [center_idx], dtype=torch.long )

    ctx_data = Data()
    
    ctx_node = torch.tensor(list(range(0, center_idx)) + list(range(center_idx+1, num_nodes)), dtype=torch.long)
    
    ctx_edge_index, _ = tg_utils.subgraph(ctx_node, data.edge_index, relabel_nodes=True)

    ctx_data.x = data.x[ctx_node].clone()
    ctx_data.edge_index = ctx_edge_index
    
    return data, ctx_data



    
def pool_func(x, batch, mode = "mean"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

criterion = nn.BCEWithLogitsLoss()

def train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device):
    model_substruct.train()
    model_context.train()

    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_substruct, batch_ctx = batch
        batch_substruct, batch_ctx = batch_substruct.to(device), batch_ctx.to(device)

        # creating substructure representation
        substruct_rep = model_substruct(batch_substruct.x, batch_substruct.edge_index)[batch_substruct.center_index]
        
        ### creating context representations
        ctx_node_rep = model_context(batch_ctx.x, batch_ctx.edge_index)

        #Contexts are represented by 
        if args.mode == "cbow":
            # positive context representation
            context_rep = pool_func(ctx_node_rep, batch_ctx.batch, mode = args.context_pooling)
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat([context_rep[cycle_index(len(context_rep), i+1)] for i in range(args.neg_samples)], dim = 0)
            
            pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
            pred_neg = torch.sum(substruct_rep.repeat((args.neg_samples, 1))*neg_context_rep, dim = 1)
        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args.neg_samples*loss_neg
        loss.backward()
        
        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5* (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))

    return balanced_loss_accum/step, acc_accum/step

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
    parser.add_argument('--csize', type=int, default=2,
                        help='context size (default: 2).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--context_pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    parser.add_argument('--mode', type=str, default = "cbow", help = "cbow")
    parser.add_argument('--dataset', type=str, default = 'COLLAB', help='root directory of dataset for pretraining')
    parser.add_argument('--pretrain_data_ratio', type=float, default=0.5)
    parser.add_argument('--output_model_dir', type=str, default = 'ckpts', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()

    # set up output directory
    args.output_model_dir = args.output_model_dir + \
        '/{}/ctxpred/'.format(args.dataset)
    print(args.output_model_dir)
    os.makedirs(args.output_model_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    print(args.mode)
    print("num layer: %d" %(args.num_layer))

    #set up dataset and transform function.
    dataset = TUDatasetExt("./datasets/" + args.dataset, args.dataset, transform = ContextExtract)
    num_node_type = dataset.num_node_type()
    if os.path.isfile('./indices/{}/pretrain_indices.pt'.format(args.dataset)):
        pretrain_indices = torch.load('./indices/{}/pretrain_indices.pt'.format(args.dataset))
        dataset = dataset[pretrain_indices]
    else:
        pretrain_indices = torch.tensor( np.random.choice(len(dataset), int(len(dataset) * args.pretrain_data_ratio), replace=False), dtype=torch.long)
        
        dataset = dataset[pretrain_indices]
        os.makedirs('./indices/{}/'.format(args.dataset))
        torch.save(pretrain_indices, './indices/{}/pretrain_indices.pt'.format(args.dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #set up models, one for pre-training and one for context embeddings
    model_substruct = GNN(args.num_layer, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    model_context = GNN(args.csize, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)

    #set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device)
        print(train_loss, train_acc)

        if not args.output_model_dir == "" and epoch % 20 == 0:
            torch.save(model_substruct.state_dict(), args.output_model_dir + "/{}.pth".format(epoch))

if __name__ == "__main__":
    main()
