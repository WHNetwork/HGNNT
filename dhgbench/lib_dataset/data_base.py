import torch
import pickle
import numpy as np
import os
import os.path as osp
from scipy.sparse import csr_matrix
from torch.utils.data import random_split
from lib_dataset.preprocessing import *
from lib_dataset.convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_geometric.data import Data

import warnings
warnings.filterwarnings('ignore')

class HyperDataset(object):
    
    def __init__(self, args):

        self.device = args.device 
        self.args=args
        self.name=args.dname 
        self.method=args.method 
        
        self.trad_list=['cora', 'citeseer', 'pubmed',
                        'coauthor_cora', 'coauthor_dblp',
                        '20newsW100', 'ModelNet40','NTU2012', 
                        'Mushroom','zoo',
                        'yelp','walmart-trips-100','house-committees-100','walmart-trips','house-committees']
        
        self.large_list=['amazon_review','magpm','magpm_mini','trivago','ogbn_mag']
        self.fair_list = ['bail','credit','german']
        self.hete_list = ['actor','amazon','pokec','twitch']
        self.hg_list = ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid",
                        "IMDB_dir_form", "IMDB_dir_genre",
                        "IMDB_wri_form", "IMDB_wri_genre",
                        "stream_player","twitter_friend"] 
        
        self.trad_path='../data/trad_data'
        self.large_path='../data/large_data/' 
        self.fair_path='../data/fair_data/'  
        self.hete_path = '../data/hete_data/'
        self.hg_path = '../data/hgcls_data/'
                    
        if any(self.name in container for container in (self.trad_list,self.fair_list,self.hete_list,self.large_list)):
            self.load_data()
        elif self.name in self.hg_list:
            self.load_multi_data()
        else:
            raise ValueError('Undefined data')

    # -------------------------- Hypergraph-level Task Dataset Loading ------------------------
    
    def load_multi_data(self):
    
        if self.name in ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid"]:
            folder = "RHG"
        elif self.name in ["stream_player"]:
            folder = "STEAM"
        elif self.name in ["IMDB_dir_form", "IMDB_dir_genre"]:
            folder = "IMDB"
        elif self.name in ["IMDB_wri_form", "IMDB_wri_genre"]:
            folder = "IMDB"
        elif self.name in ["twitter_friend"]:
            folder = "TWITTER"
        else:
            raise NotImplementedError

        # read data
        x_list = []
        with open(f"{self.hg_path}/{folder}/{self.name}.txt", "r") as f:
            n_g = int(f.readline().strip())
            for _ in range(n_g):
                
                row = f.readline().strip().split()
                num_v, num_e = int(row[0]), int(row[1])
                g_lbl = [int(x) for x in row[2:]]
                v_lbl = f.readline().strip().split()
                v_lbl = [[int(x) for x in s.split("/")] for s in v_lbl]
                e_list = []
                for _ in range(num_e):#
                    row = f.readline().strip().split()
                    e_list.append([int(x) for x in row])
                
                hyperedge_index = convert_to_hyperedge_index(e_list) 
                v_deg=compute_degree_list(hyperedge_index)
                
                x_list.append(
                        {
                            "num_v": num_v,
                            "num_e": num_e,
                            "v_lbl": v_deg,
                            "g_lbl": g_lbl,
                            "edge_index": hyperedge_index,
                        }
                    )        

            v_lbl_set, g_lbl_set = set(), set()
            for x in x_list:
                if isinstance(x["v_lbl"][0], list):
                    for v_lbl in x["v_lbl"]:
                        v_lbl_set.update(v_lbl)
                else:
                    v_lbl_set.update(x["v_lbl"])
                g_lbl_set.update(x["g_lbl"])
                
            v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
            g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}
            ft_dim, n_classes = len(v_lbl_set), len(g_lbl_set)
            for x in x_list:
                x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
                if isinstance(x["v_lbl"][0], list):
                    x["v_lbl"] = [tuple(sorted([v_lbl_map[c] for c in s])) for s in x["v_lbl"]]
                else:
                    x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]
                x["v_ft"] = np.zeros((x["num_v"], ft_dim))
                row_idx, col_idx = [], []
                for v_idx, v_lbls in enumerate(x["v_lbl"]):
                    if isinstance(v_lbls, list) or isinstance(v_lbls, tuple):
                        for v_lbl in v_lbls:
                            row_idx.append(v_idx)
                            col_idx.append(v_lbl)
                    else:
                        row_idx.append(v_idx)
                        col_idx.append(v_lbls)
                x["v_ft"][row_idx, col_idx] = 1
            y_list = []
            y_list = [g["g_lbl"][0] for g in x_list]
            
        hgs=[] 
        for i,hg in enumerate(x_list):
            hypergraph = Data(x = torch.FloatTensor(hg["v_ft"]), hyperedge_index = hg["edge_index"], y = torch.LongTensor([y_list[i]]),
                            num_nodes = hg['num_v'],num_edges = hg['num_e'])
            hgs.append(hypergraph)
            
        self.multi_hypergraphs = MultiHyperaphDatasets(x=hgs, y=torch.LongTensor(y_list), 
                                                    num_hypergraphs=n_g, num_features=ft_dim, num_classes=n_classes)
    
    # -------------------------- Hypernode/edge-level Task Dataset Loading ------------------------
    
    def load_data(self):
        
        dname=self.name
        self.sens=None 
        self.norm=None 
        
        if dname in self.trad_list:
            
            path=self.trad_path
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = osp.join(path, 'cocitation','')
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = osp.join(path, 'coauthorship','')
            elif dname in ['yelp']:
                p2raw = osp.join(path, 'yelp','')
            else:
                p2raw = path
            
            f_noise = self.args.feature_noise
            if (f_noise is not None) and dname in ['walmart-trips-100','house-committees-100','walmart-trips', 'house-committees']:
                dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',p2raw = p2raw,
                                            feature_noise=f_noise)
            else:
                dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',p2raw = p2raw)
                
        elif dname in self.fair_list:
            
            p2raw = self.fair_path
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                            p2raw = p2raw)
            if dname=="credit":
                sens_idx = 1
            else:
                sens_idx = 0
            self.sens = dataset.data.x[:,sens_idx] 
        
        elif dname in self.hete_list:

            p2raw = self.hete_path
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                            p2raw = p2raw)

        elif dname in self.large_list:
        
            p2raw = self.large_path
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                            p2raw = p2raw)
        
        else:
            raise ValueError('Undefined data')
        
        data = dataset.data
        
        if dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100','zoo']:
            #   Shift the y label to start with 0
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x+1])

        self.data=data 
        
        self.x=data.x
        self.hyperedge_index=data.edge_index
        self.y=data.y

    def _initialization_(self):
        
        self.num_classes= len(self.y.unique())
        self.num_features=self.x.shape[1]
        self.num_nodes=self.x.shape[0]
        declared_edges = getattr(self, 'hirseqgru_num_hyperedges', None)
        if self.method == 'HIRCall':
            declared_edges = getattr(self, 'hircall_num_hyperedges', None)
        self.num_hyperedges=(declared_edges if declared_edges is not None
                            else len(self.hyperedge_index[1].unique()))
 
    def calculate_memory_size(self):

        labels_size = self.y.numel() * self.y.element_size()  
        features_size = self.x.numel() * self.x.element_size()
        hyperedge_index_size = self.hyperedge_index.numel() * self.hyperedge_index.element_size()

        total_size_bytes = labels_size + features_size + hyperedge_index_size
        total_size_gb = total_size_bytes / (1024 ** 3)  

        print(f"The memory cost of hypergraph: {total_size_bytes} Byte")
        print(f"The memory cost of hypergraph: {total_size_gb:.2f} GB")

    def to(self, device: str):
        
        self.x = self.x.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.y = self.y.to(device)
            
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed = None):

        num_train = int(self.num_nodes * train_ratio)
        num_val = int(self.num_nodes * val_ratio)
        num_test = self.num_nodes - (num_train + num_val)

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.default_generator

        train_set, val_set, test_set = random_split(
            torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
            generator=generator)
        train_idx, val_idx, test_idx = \
            train_set.indices, val_set.indices, test_set.indices
        train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
        val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
        test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        '''
        [tensor([False, False, False,  ..., False, False, False], device='cuda:0'),
        tensor([False, False,  True,  ..., False,  True,  True], device='cuda:0'),
        tensor([ True,  True, False,  ...,  True, False, False], device='cuda:0')]
        '''
        return {'train':train_mask, 'valid': val_mask, 'test': test_mask}
        
        # return [train_mask, val_mask, test_mask]

class MultiHyperaphDatasets:
    def __init__(self, x, y, num_hypergraphs, num_features, num_classes): 

        self.x = x 
        self.y = y
        self.num_hypergraphs = num_hypergraphs
        self.num_features = num_features
        self.num_classes = num_classes

    def __len__(self):
        return self.num_hypergraphs 

    def __getitem__(self, idx):

        return self.x[idx]

