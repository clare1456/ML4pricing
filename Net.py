'''
File: Net.py
Project: ML4pricing
File Created: Sunday, 16th April 2023 3:03:28 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from utils import TestArgs

class GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "GAT"
        self.args = args
        # build network 
        self.hidden_dim = 256
        self.heads = 8
        self.dropout = 0.2
        self.node_linear = nn.Linear(args.node_feature_dim, self.hidden_dim)
        self.column_linear = nn.Linear(args.column_feature_dim, self.hidden_dim)
        self.conv1 = GATConv(self.hidden_dim, self.hidden_dim, heads=self.heads, dropout=self.dropout) # node to column
        self.conv2 = GATConv(self.hidden_dim*self.heads, self.hidden_dim, dropout=self.dropout) # column to node
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim//2, 1)
        )

    def forward(self, node_features, column_features, edges):
        # transfer to tensor if type is not tensor
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float)
        if not isinstance(column_features, torch.Tensor):
            column_features = torch.tensor(column_features, dtype=torch.float)
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges, dtype=torch.long)
        # network calculation 
        node_embeddings = F.relu(self.node_linear(node_features))
        column_embeddings = F.relu(self.column_linear(column_features))
        # embedding concat
        embeddings = torch.cat([node_embeddings, column_embeddings], dim=0)
        embeddings = F.relu(self.conv1(embeddings, edges)) # node to column
        embeddings = F.relu(self.conv2(embeddings, torch.flip(edges, [1]))) # column to node
        logits = self.output(embeddings[:self.args.node_num]) # get node logits
        return logits.squeeze(1)

    def save_model(self, path):
        torch.save(self.state_dict(), path + self.name + '.pth')

    def load_model(self, path):
        torch.load(path + self.name + '.pth')
        
if __name__ == "__main__":
    # build model
    args = TestArgs()
    model = GAT(args)
    # create random test data
    node_features = torch.randn(args.node_num, args.node_feature_dim)
    column_num = 200
    column_features = torch.randn(column_num, args.column_feature_dim)
    edges = [[], []]
    for ci in range(column_num):
        path_length = np.random.randint(2, args.node_num//5)
        path = np.random.choice(args.node_num, path_length, replace=False)
        path = [0] + list(path)
        column_features[ci][3:] = 0
        for ni in path:
            column_features[ni] = 1
            edges[0].append(ni)
            edges[1].append(ci)
    edges = np.array(edges)
    # model predict
    offsets_pred = model(node_features, column_features, edges)
    print(offsets_pred.shape)
            
