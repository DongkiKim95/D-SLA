import argparse

from tu_dataset import TUDatasetExt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import DataLoader

from tqdm import tqdm
import numpy as np

from model import GNN

from splitters import random_split
import os

from sklearn.metrics import average_precision_score, roc_auc_score

criterion = nn.BCEWithLogitsLoss()

class EdgePairExtract():
    def __init__(self, num_pair):
        self.num_pair = num_pair

    def __call__(self, data):
        num_nodes = data.num_nodes

        not_ego_indices = torch.tensor( [i for i in range(num_nodes) if i not in data.ego_indices], dtype=torch.long)
        
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[not_ego_indices] = 1
        
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        num_edges = edge_mask.sum().to(torch.long)
        pos_edge_candidates = edge_mask.nonzero(as_tuple=False).flatten()

        if self.num_pair >= num_edges:
            pos_edge_idx = pos_edge_candidates
        else:
            pos_edge_idx = np.random.choice(pos_edge_candidates, self.num_pair, replace=False)

        data.pos_edge_index = data.edge_index[:, pos_edge_idx]

        adj = torch.ones((num_nodes, num_nodes), dtype=torch.long)
        adj[data.edge_index[0], data.edge_index[1]] = 0
        adj[range(num_nodes), range(num_nodes)] = 0

        neg_edge_candidates = adj.nonzero(as_tuple=False).t()
        neg_edge_num = neg_edge_candidates.size(1)
        if self.num_pair >= neg_edge_num:
            data.neg_edge_index = neg_edge_candidates
        else:
            neg_edge_idx = np.random.choice(neg_edge_candidates.size(1), self.num_pair, replace=False)
            data.neg_edge_index = neg_edge_candidates[:, neg_edge_idx]
            
        return data

def train(args, model, device, loader, optimizer, epoch):
    model.train()

    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Train")):

        batch = batch.to(device)
        node_emb = model(batch.x, batch.edge_index)

        positive_score = torch.sum(node_emb[batch.pos_edge_index[0]] * node_emb[batch.pos_edge_index[1]], dim = 1)
        negative_score = torch.sum(node_emb[batch.neg_edge_index[0]] * node_emb[batch.neg_edge_index[1]], dim = 1)

        optimizer.zero_grad()
        loss = criterion(positive_score, torch.ones_like(positive_score)) + criterion(negative_score, torch.zeros_like(negative_score))
        loss.backward()
        optimizer.step()
        

        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum/step

def test(args, model, device, loader, valid):
    model.eval()

    correct_accum = 0
    whole_accum = 0
    roc_list = []
    ap_list = []
    if valid:
        valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Valid' if valid else "Test"):
            batch = batch.to(device)

            node_emb = model(batch.x, batch.edge_index)

            positive_score = torch.sum(node_emb[batch.pos_edge_index[0]] * node_emb[batch.pos_edge_index[1]], dim = 1)
            negative_score = torch.sum(node_emb[batch.neg_edge_index[0]] * node_emb[batch.neg_edge_index[1]], dim = 1)

            positive_score = torch.sigmoid(positive_score)
            negative_score = torch.sigmoid(negative_score)

            positive_label = torch.ones_like(positive_score)
            negative_label = torch.zeros_like(negative_score)

            score = torch.cat([positive_score, negative_score], dim=0).cpu()
            label = torch.cat([positive_label, negative_label], dim=0).cpu()

            ap = average_precision_score(label, score)
            ap_list.append(ap)

            if valid:
                loss = criterion(positive_score, torch.ones_like(positive_score)) + criterion(negative_score, torch.zeros_like(negative_score))
                valid_loss += loss.cpu().item()

        ap = sum(ap_list) / len(ap_list)
        
    if valid:
        valid_loss = valid_loss / len(loader)
        return valid_loss, ap
        
    return ap


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
    parser.add_argument('--model_file', type=str, default = '')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--num_workers', type=int, default = 8)
    parser.add_argument('--num_pair', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #set up dataset
    dataset = TUDatasetExt("./datasets/" + args.dataset, args.dataset)
    num_node_type = dataset.num_node_type()
    pretrain_indices = torch.load('./indices/{}/pretrain_indices.pt'.format(args.dataset))
    finetune_indices = torch.tensor([i for i in range(len(dataset)) if i not in pretrain_indices], dtype=torch.long)
    dataset = dataset[finetune_indices]
    print(dataset)
    
    train_dataset, valid_dataset, test_dataset = \
        random_split(dataset, frac_train=0.2, frac_valid=0.2, frac_test=0.6, seed=args.seed)
    train_dataset.transform = EdgePairExtract(args.num_pair)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    #set up model
    model = GNN(args.num_layer, args.emb_dim, num_node_type, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    if args.model_file != '':
        model.load_state_dict(torch.load(args.model_file))
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)   
    print(optimizer)


    if args.model_file != '':
        path = os.path.join(args.model_file)
        model_name = str(os.path.basename(os.path.dirname(path)))
    else:  
        model_name = 'no_pretrain'
    args.result_path = 'results' + '/' + args.dataset + '/' + model_name + '/' + str(args.seed) + '/'
    print(args.result_path)
    os.makedirs(args.result_path)

    valid_ap, test_ap = [], []
    
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        
        valid_loss, valid_ap_ = test(args, model, device, valid_loader, valid=True)
        valid_ap.append(valid_ap_)

        test_ap_ = test(args, model, device, test_loader, valid=False)
        test_ap.append(test_ap_)

        print(train_loss, valid_loss, valid_ap_, test_ap_)

    valid_ap = torch.tensor(valid_ap)
    test_ap = torch.tensor(test_ap)
    valid_max_idx = valid_ap.argmax()

    print(args.seed, valid_ap[valid_max_idx].item(), test_ap[valid_max_idx].item())
    with open(args.result_path + '/result.txt', 'w') as f:
        f.write("{} {} {}\n".format(args.seed, valid_ap[valid_max_idx], test_ap[valid_max_idx]))
    

if __name__ == "__main__":
    main()
