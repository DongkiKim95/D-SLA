import os
import torch
import random
import networkx as nx
import pandas as pd
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from torch_geometric.utils import k_hop_subgraph
from itertools import repeat, product, chain
from collections import Counter, deque
from networkx.algorithms.traversal.breadth_first_search import generic_bfs_edges

import torch_geometric.utils as tg_utils
import networkx as nx


def nx_to_graph_data_obj(g, center_id, allowable_features_downstream=None,
                         allowable_features_pretrain=None,
                         node_id_to_go_labels=None):
    """
    Converts nx graph of PPI to pytorch geometric Data object.
    :param g: nx graph object of ego graph
    :param center_id: node id of center node in the ego graph
    :param allowable_features_downstream: list of possible go function node
    features for the downstream task. The resulting go_target_downstream node
    feature vector will be in this order.
    :param allowable_features_pretrain: list of possible go function node
    features for the pretraining task. The resulting go_target_pretrain node
    feature vector will be in this order.
    :param node_id_to_go_labels: dict that maps node id to a list of its
    corresponding go labels
    :return: pytorch geometric Data object with the following attributes:
    edge_attr
    edge_index
    x
    species_id
    center_node_idx
    go_target_downstream (only if node_id_to_go_labels is not None)
    go_target_pretrain (only if node_id_to_go_labels is not None)
    """
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # in a particular ordering. Will be used as a mapping to convert
    # between nx node ids and data obj node indices

    x = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
    # we don't have any node labels, so set to dummy 1. dim n_nodes x 1

    center_node_idx = nx_node_ids.index(center_id)
    center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)

    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        edge_feature = [attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
                        attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
                        attr_dict['w7'], 0, 0]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)

    try:
        species_id = int(nx_node_ids[0].split('.')[0])  # nx node id is of the form:
        # species_id.protein_id
        species_id = torch.tensor([species_id], dtype=torch.long)
    except:  # occurs when nx node id has no species id info. For the extract
        # substructure context pair transform, where we convert a data obj to
        # a nx graph obj (which does not have original node id info)
        species_id = torch.tensor([0], dtype=torch.long)    # dummy species
        # id is 0

    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.species_id = species_id
    data.center_node_idx = center_node_idx

    if node_id_to_go_labels:  # supervised case with go node labels
        # Construct a dim n_pretrain_go_classes tensor and a
        # n_downstream_go_classes tensor for the center node. 0 is no data
        # or negative, 1 is positive.
        downstream_go_node_feature = [0] * len(allowable_features_downstream)
        pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
        if center_id in node_id_to_go_labels:
            go_labels = node_id_to_go_labels[center_id]
            # get indices of allowable_features_downstream that match with elements
            # in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
                                        dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
                                        dtype=torch.long)

    return data

def graph_data_obj_to_nx(data):
    """
    Converts pytorch geometric Data obj to network x data object.
    :param data: pytorch geometric Data object
    :return: nx graph object
    """
    G = nx.Graph()

    # edges
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(0, n_edges, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        w1, w2, w3, w4, w5, w6, w7, _, _ = edge_attr[j].astype(bool)
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5,
                       w6=w6, w7=w7)

    # # add center node id information in final nx graph object
    # nx.set_node_attributes(G, {data.center_node_idx.item(): True}, 'is_centre')

    return G


class BioDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_type,
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: the data directory that contains a raw and processed dir
        :param data_type: either supervised or unsupervised
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        self.root = root
        self.data_type = data_type

        super(BioDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #raise NotImplementedError('Data is assumed to be processed')
        if self.data_type == 'supervised': # 8 labelled species
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090', '4932', '7955']
        else: # unsupervised: 8 labelled species, and 42 top unlabelled species by n_nodes.
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090',
            '4932', '7955', '3694', '39947', '10116', '443255', '9913', '13616',
            '3847', '4577', '8364', '9823', '9615', '9544', '9796', '3055', '7159',
            '9031', '7739', '395019', '88036', '9685', '9258', '9598', '485913',
            '44689', '9593', '7897', '31033', '749414', '59729', '536227', '4081',
            '8090', '9601', '749927', '13735', '448385', '457427', '3711', '479433',
            '479432', '28377', '9646']
        return file_name_list


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        raise NotImplementedError('Data is assumed to be processed')


class BioDataset_candidate(BioDataset):
    def __init__(self,
                 root,
                 data_type,
                 edge_pert_strength,
                 mask_strength,
                 k,
                 num_candidates,
                 increase_pert,
                 return_strong_pert,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        
        self.edge_pert_strength = edge_pert_strength
        self.mask_strength = mask_strength
        # Get subgraph could be random walk or simple k-hop neighbor subgraph.
        self.k = k
        self.num_candidates = num_candidates
        self.increase_pert = increase_pert
        self.return_strong_pert = return_strong_pert
        self.generated_edge_attr_prob = torch.tensor([0.0322, 0.0559, 0.0827, 0.0752, 0.0406, 0.7208, 0.0428, 0.0000, 0.0000])
        
        super(BioDataset_candidate, self).__init__(root, data_type, empty, transform, pre_transform, pre_filter)

    def subgraph_edge_pert(self, data, subset, sub_edge_index, edge_mask, i):
        sub_edge_attr = data.edge_attr[edge_mask]

        edge_index = data.edge_index[:,~edge_mask]
        edge_attr = data.edge_attr[~edge_mask]

        _, sub_edge_num = sub_edge_index.size()
        sub_edge_num = int(sub_edge_num / 2)
        pert_num = int(sub_edge_num * self.edge_pert_strength * i)
        if pert_num == 0:
            pert_num = 1

        # delete edges
        idx_chosen = np.random.choice(sub_edge_num, (sub_edge_num - pert_num), replace=False)
        idx_chosen = np.concatenate([idx_chosen*2, idx_chosen*2+1])

        sub_edge_index = sub_edge_index[:, idx_chosen]
        
        sub_edge_attr = sub_edge_attr[idx_chosen]
        
        # add edges
        node_num = data.x.size(0)
        adj = torch.ones((node_num, node_num))
        adj[range(node_num), range(node_num)] = 0
        
        not_subset = np.setdiff1d(np.arange(node_num), subset)
        adj[not_subset,:] = 0
        adj[:, not_subset] = 0
        adj[sub_edge_index[0], sub_edge_index[1]] = 0
        adj = torch.tril(adj, diagonal=0)
        edge_index_nonexist = adj.nonzero(as_tuple=False).t()

        idx_add = np.random.choice(edge_index_nonexist.shape[1], pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]
        edge_index_add = torch.cat([edge_index_add, edge_index_add[[1,0],:]], dim=1)

        # Generate edge attributes. The attribute of an edge should not be a zero vector
        edge_attr_add = torch.bernoulli(self.generated_edge_attr_prob.repeat(pert_num, 1))
        if (edge_attr_add.sum(dim=-1) == 0).any():
            idxs = (edge_attr_add.sum(dim=-1) == 0).nonzero(as_tuple=False).flatten()
            edge_attr_add[idxs, 5] = 1
            
        edge_attr_add = torch.cat([edge_attr_add, edge_attr_add], dim=0)

        edge_index = torch.cat((edge_index, sub_edge_index, edge_index_add), dim=1)
        edge_attr = torch.cat((edge_attr, sub_edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        
        return data, pert_num * 2

    def get(self, idx):
        candidates = [Data() for _ in range(self.num_candidates)]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[candidates[0].__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])

            for i in range(self.num_candidates):
                candidates[i][key] = item[s]

        # Sampling subgraph
        data = candidates[0]
        node_num, _ = data.x.size()
        center_idx = torch.tensor(np.random.choice(node_num, 1, replace=False))
        subset, sub_edge_index, _, edge_mask = k_hop_subgraph(center_idx, self.k, data.edge_index, num_nodes=node_num)

        for i in range(self.num_candidates):
            # Strongly perturbed graph
            if self.return_strong_pert and i == self.num_candidates - 1:
                all_set = torch.tensor(range(node_num), dtype=torch.long)
                edge_index = candidates[i].edge_index
                edge_mask = torch.ones((edge_index.size(1), ) ,dtype=torch.bool)

                candidates[i], dist = self.subgraph_edge_pert(candidates[i], all_set, edge_index, edge_mask, 4)
                candidates[i].dist = dist
                break
                
            if i > 0:
                if self.increase_pert:
                    candidates[i], dist = self.subgraph_edge_pert(candidates[i], subset, sub_edge_index, edge_mask, i)
                else:
                    candidates[i], dist = self.subgraph_edge_pert(candidates[i], subset, sub_edge_index, edge_mask, 1)
            else:
                dist = 0
            candidates[i].dist = dist


        return candidates

def list_collate_fn(data_list):
    num_candidates = len(data_list[0])
    
    data_full = []
    for d in data_list:
        data_full.extend(d)

    batch = Batch.from_data_list(data_full)
    
    return batch



if __name__ == "__main__":
    sup_dataset = BioDataset("dataset/" + 'supervised', data_type='supervised')
    print(sup_dataset)

    unsup_dataset = BioDataset('dataset/unsupervised', data_type='unsupervised')
    print(unsup_dataset)