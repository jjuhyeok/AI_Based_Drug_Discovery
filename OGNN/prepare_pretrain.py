import pandas as pd
import torch, os
from tqdm import tqdm
from dualgraph.mol import smiles2graphwithface
from dualgraph.dataset import DGData
import numpy as np

def get_graph(smiles):
    data = DGData()

    graph = smiles2graphwithface(smiles)

    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
    data.num_rings = int(graph["num_rings"])
    data.n_edges = int(graph["n_edges"])
    data.n_nodes = int(graph["n_nodes"])
    data.n_nfs = int(graph["n_nfs"])
    
    return data

data = pd.read_csv('pretrain/zinc_combined_apr_8_2019.csv', index_col=None)
data = data[['zinc_id', 'smiles']]
data.columns = ['sid', 'smiles']
data['dataset'] = 'zinc_combined_apr_8_2019'
for csv in tqdm(os.listdir( 'pretrain/')):
    if 'zinc' in csv:
        continue
    df = pd.read_csv(f'pretrain/{csv}', index_col=None)                    
    df['dataset'] = csv[:-4]
    df['sid'] = csv[:-4] + '_' +  df.reset_index()['index'].astype(str)
    if csv[:-4] == 'bace':
        df = df[['sid', 'dataset', 'mol']]
        df.columns = ['sid', 'dataset', 'smiles']
    else:
        df = df[['sid', 'dataset', 'smiles']]
    data = pd.concat([data, df]).reset_index(drop=True)

if not os.path.exists('graph_pt'):
    os.mkdir('graph_pt')
for smile, zid in tqdm(data[['smiles', 'sid']].values):
    graph = get_graph(smile)
    torch.save(graph, f'graph_pt/{zid}.pt')    
