import os
import os.path as osp
import shutil
from itertools import repeat

import numpy as np
import torch
import torch_geometric.utils as tg_utils
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from tqdm import tqdm

# Add degree for node attributes
def deg_transform(data):
    num_nodes = data.num_nodes

    row, col = data.edge_index
    deg = tg_utils.degree(row, num_nodes).to(torch.long)
    deg = deg.view((-1, 1))

    data.x = deg

    return data

# tudataset adopted from torch_geometric==1.1.0
class TUDatasetExt(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=deg_transform,
                 pre_filter=None,
                 processed_filename='data_edge_deg.pt'):
        self.name = name
        self.processed_filename = processed_filename
        self.num_pair = 10
        super(TUDatasetExt, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename

    def num_node_type(self):
        return self.data.x.max()+1

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        # Remove cliques
        data_list = []
        for idx in range(len(self)):
            data = self.get(idx)
            num_nodes = data.edge_index.max() + 1
            data.num_nodes = num_nodes
            num_edges = data.edge_index.size(1)
            
            if num_nodes * (num_nodes - 1) != num_edges: 
                data_list.append(data)
        
        for data in tqdm(data_list):
            num_nodes = data.num_nodes

            # Add ego node indices
            ego_indices = []
            for i in range(num_nodes):
                subset, _, _, _ = tg_utils.k_hop_subgraph([i], 1, data.edge_index, flow='target_to_source')
                if len(subset) == num_nodes:
                    ego_indices.append(i)
            data.ego_indices = torch.tensor(ego_indices, dtype=torch.long)
            
            not_ego_indices = torch.tensor( [i for i in range(num_nodes) if i not in data.ego_indices], dtype=torch.long)
            
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[not_ego_indices] = 1
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            num_edges = edge_mask.sum().to(torch.long)
            pos_edge_candidates = edge_mask.nonzero(as_tuple=False).flatten()

            # Extract edges for validation and test
            pred_number = 10
            if pred_number >= num_edges:
                pos_edge_idx = pos_edge_candidates
            else:
                pos_edge_idx = np.random.choice(pos_edge_candidates, pred_number, replace=False)
        
            data.pos_edge_index = data.edge_index[:, pos_edge_idx]

            adj = torch.ones((num_nodes, num_nodes), dtype=torch.long)
            adj[data.edge_index[0], data.edge_index[1]] = 0
            adj[range(num_nodes), range(num_nodes)] = 0

            neg_edge_candidates = adj.nonzero(as_tuple=False).t()
            neg_edge_num = neg_edge_candidates.size(1)
            if pred_number >= neg_edge_num:
                data.neg_edge_index = neg_edge_candidates
            else:
                neg_edge_idx = np.random.choice(neg_edge_candidates.size(1), pred_number, replace=False)
                data.neg_edge_index = neg_edge_candidates[:, neg_edge_idx]

        self.data, self.slices = self.collate(data_list)
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]
        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

if __name__ == '__main__':
    dataset = TUDatasetExt('./datasets/COLLAB', 'COLLAB')

    for data in tqdm(dataset):
        pos_edge_index = data.pos_edge_index
        neg_edge_index = data.neg_edge_index
        
        edge_index = data.edge_index
        for i in range(pos_edge_index.size(1)):
            u = pos_edge_index[0,i]
            v = pos_edge_index[1,i]

            assert ((edge_index[0] == u) & (edge_index[1] == v)).sum() == 1

        for i in range(neg_edge_index.size(1)):
            u = neg_edge_index[0,i]
            v = neg_edge_index[1,i]

            assert ((edge_index[0] == u) & (edge_index[1] == v)).sum() == 0