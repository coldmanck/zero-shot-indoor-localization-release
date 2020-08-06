import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import os.path as osp
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import f1_score

from torch.utils.data import Dataset, DataLoader
import pickle, os
import numpy as np
from tqdm import tqdm

import torch.optim as optim
from torchvision import transforms


class MapDataset(Dataset):
    def __init__(self, args):
        with open('data/' + args.dataset + '/loc_vec.npy', 'rb') as f:
            self.loc_vec = np.load(f)[1:, :]
    
    def __len__(self):
        return len(self.loc_vec)
    
    def __getitem__(self, idx):
        return torch.FloatTensor((self.loc_vec[idx])), torch.LongTensor((idx,))


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


def train():
    model.train()

    loss_all = 0
    for data, label in train_loader:
        data, label = data.to(device), label.view(-1).to(device)
        map_edge_index = torch.tensor([start, end], dtype=torch.long).to(device)
        optimizer.zero_grad()
        output, logits = model(data, map_edge_index)

        loss = criterion(output, label)
        loss.backward()
        loss_all += (loss.item() / label.shape[0])
        optimizer.step()
        
    return loss_all / len(train_loader)

def evaluate(loader):
    model.eval()

    loss_all = 0
    accuracy_all = 0
    print('Evaulating...')
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.view(-1).to(device)
            map_edge_index = torch.tensor([start, end], dtype=torch.long).to(device)
            output, logits = model(data, map_edge_index)
            loss = criterion(output, label)
            loss_all += (loss.item() / label.shape[0])
            
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == label.view(*top_class.shape)
#             print('prediction:', top_class.data.reshape(-1), 'gt:', label.data.reshape(-1))
            accuracy_all += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return loss_all / len(loader), accuracy_all / len(loader)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', dest='dataset', type=str, default='icube', choices=['icube', 'wcp'], help='dataset to be trained on')
args = parser.parse_args()

train_dataset = MapDataset(args)
batch_size = 214 if args.dataset == 'icube' else 394
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
print('len of training set:', len(train_dataset), ', len of training loader:', len(train_loader))

with open('data/' + args.dataset + '/adjacency_matrix.npy', 'rb') as f:
    adj_matrix = np.load(f)[1:, 1:]
start = []
end = []
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if adj_matrix[i][j] != 0:
            start.append(i)
            end.append(j)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MapGraphNet(args)
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10000
best_train_acc = 0.0

train_losses = []
train_accs = []
for epoch in range(epochs):
    print(f'Training epoch {epoch}...')
    loss = train()
    print(f'Epoch: {epoch:03d}, Training Loss: {loss:.5f}')
    
    if (epoch+1) % 1 == 0:
        print(f'Evaluating on training set, epoch {epoch}...')
        train_loss, train_acc = evaluate(train_loader)
        print(f'Epoch: {epoch:03d}, Training Loss: {train_loss:.5f}, Accuracy: {train_acc:.5f}')

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            
            save_addr = f'checkpoints_{args.dataset}/loc_vec-gcn-best_model-epoch{epoch}.pth'

            print(f'Saving best model as ...')
            checkpoint = {'state_dict': model.state_dict(), 
                          'best_train_accuracy': best_train_acc,
                          'train_loss': train_loss,
                          'nb_of_epoch': epoch}
            torch.save(checkpoint, save_addr)


# extract Map2Vec embedding
ckpt = torch.load(save_addr)
print(ckpt['best_train_accuracy'])

model.load_state_dict(torch.load('checkpoints_wcp/loc_vec-gcn-best_model-epoch8761.pth')['state_dict'])

model.eval()
with torch.no_grad():
    for data, label in train_loader:
        data, label = data.to(device), label.view(-1).to(device)
        map_edge_index = torch.tensor([start, end], dtype=torch.long).to(device)
        output, logits = model(data, map_edge_index)
        
# Save embedding
if args.dataset == 'icube':
    save_addr = 'data/icube/loc_vec_trained_214.npy'
else:
    save_addr = 'data/wcp/loc_vec_trained_394.npy'
with open(save_addr, 'wb') as f:
    np.save(f, logits.cpu().numpy())
    
