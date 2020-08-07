import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch_geometric.nn import GCNConv, GATConv

import pickle, os
import numpy as np
from tqdm import tqdm

from lib.datasets import MapDataset
from lib.models import MapGraphNet

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
            accuracy_all += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return loss_all / len(loader), accuracy_all / len(loader)

if __name__ == '__main__':
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
    best_epoch = 0
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
                best_epoch = epoch
                
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
    model.load_state_dict(torch.load(f'checkpoints_{args.dataset}/loc_vec-gcn-best_model-epoch{best_epoch}.pth')['state_dict'])

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
