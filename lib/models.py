import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv

class LocationNet(nn.Module):
    def __init__(self, args, zero_shot=False):
        super().__init__()

        self.zero_shot = zero_shot
        
        self.cnn_model = torchvision.models.resnet152(pretrained=True)
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x):
                return x
        
        self.cnn_model.fc = Identity()
        
        self.location_graph_net = LocationGraphNet(args, zero_shot)
        
    def forward(self, x, edge_index):
        x = self.cnn_model(x.view(-1, 3, 224, 224)).view(-1, 4, 2048)
        
        data_list = [torch_geometric.data.Data(x=x[idx], edge_index=edge_index) for idx in range(x.shape[0])]
        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        data = iter(loader).next()
        
        x = self.location_graph_net(data)
        if self.zero_shot:
            x = F.relu(x)
            x = torch.cat([x[i].repeat(4, 1) for i in range(data.num_graphs)])
        
        return x


class LocationGraphNet(nn.Module):
    def __init__(self, args, zero_shot=False):
        super().__init__()
        self.zero_shot = zero_shot

        if args.network == 'gat': # 'gat
            self.conv1 = GATConv(2048, 256)
        else: # 'gcn' or 'baseline'
            self.conv1 = GCNConv(2048, 256)
        
        self.bn1 = nn.BatchNorm1d(256)
        
        if zero_shot and args.network == 'baseline':
            self.fc1 = nn.Linear(256*4, 2)
        elif args.dataset == 'icube':
            self.fc1 = nn.Linear(256*4, 214)
        else: # wcp
            self.fc1 = nn.Linear(256*4, 394)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)

        x = torch.cat([x[i*4:(i+1)*4].view(1, -1) for i in range(data.num_graphs)])
        x = self.fc1(x)

        if not self.zero_shot:
            x = torch.cat([x[i].repeat(4, 1) for i in range(data.num_graphs)])
            
        return x if self.zero_shot else F.log_softmax(x, dim=1)


class MapGraphNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        output_dim = 214 if args.dataset == 'icube' else 394

        self.conv1 = GCNConv(2, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc1 = nn.Linear(output_dim, output_dim)
        
    def forward(self, x, edge_index, dist_max=None, dist_argmax=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=1), x
