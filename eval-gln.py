import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from torch_geometric.nn import GCNConv, GATConv

import os.path as osp
import pickle, os
import numpy as np
import argparse
from tqdm import tqdm

from lib.models import LocationNet, LocationGraphNet
from lib.datasets import IndoorDataset

def evaluate(loader, len_of_dataset, args, eval_dist=True):
    if args.dataset == 'icube':
        loc_vec = 'data/icube/loc_vec.npy'
        n_classes = 214
    else:
        loc_vec = 'data/wcp/loc_vec.npy'
        n_classes = 394
    with open(loc_vec, 'rb') as f:
        loc_coord = np.load(f)[1:]
    
    model.eval()
    
    accuracy_all = 0
    top1_accuracy_all = 0
    top2_accuracy_all = 0
    top3_accuracy_all = 0
    top5_accuracy_all = 0
    top10_accuracy_all = 0
    
    top1_dists = 0.0
    top2_dists = 0.0
    top3_dists = 0.0
    top5_dists = 0.0
    top10_dists = 0.0
    
    # draw plots
    record_length = 30
    top1_count = [0] * record_length
    top2_count = [0] * record_length
    top3_count = [0] * record_length
    top5_count = [0] * record_length
    top10_count = [0] * record_length
    
    nan_count = 0
    iters = 0
    loss_all = 0
    
    print('Evaulating...')
    with torch.no_grad():
        for data, label in tqdm(loader):
            iters += 1
            data, label = data.to(device), label.view(-1).to(device)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                                       [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long).to(device)
            output = model(data, edge_index)
            loss = criterion(output, label)
            loss_all += loss.item()
            
            ps = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            
            equals = top_class == label.view(*top_class.shape)
            accuracy_all += torch.mean(equals.type(torch.FloatTensor)).item() * label.shape[0]

            if eval_dist:
                top_p, top_class = output.topk(10)
                top1 = []
                top2 = []
                top3 = []
                top5 = []
                top10 = []
                for i in range(top_class.shape[0]):
                    if any(top_class[i] > n_classes - 1) or any(top_class[i] < 0):
                        nan_count += 1
                        print(f'Nan result detected! Count: {nan_count}')
                        top_class[i] = torch.randint(0, n_classes, top_class[i].shape).to(device)

                    top1 += [1] if label[i] == top_class[i][0] else [0]
                    top2 += [1] if label[i] in top_class[i][:2] else [0]
                    top3 += [1] if label[i] in top_class[i][:3] else [0]
                    top5 += [1] if label[i] in top_class[i][:5] else [0]
                    top10 += [1] if label[i] in top_class[i][:10] else [0]

                    top1_dist = np.linalg.norm(loc_coord[label[i]] - loc_coord[top_class[i][0].cpu().numpy()])
                    top2_dist = sum([np.linalg.norm(loc_coord[label[i]] - loc_coord[top_class[i][k].cpu().numpy()]) for k in range(0, 2)]) / 2
                    top3_dist = sum([np.linalg.norm(loc_coord[label[i]] - loc_coord[top_class[i][k].cpu().numpy()]) for k in range(0, 3)]) / 3
                    top5_dist = sum([np.linalg.norm(loc_coord[label[i]] - loc_coord[top_class[i][k].cpu().numpy()]) for k in range(0, 5)]) / 5
                    top10_dist = sum([np.linalg.norm(loc_coord[label[i]] - loc_coord[top_class[i][k].cpu().numpy()]) for k in range(0, 10)]) / 10

                    for dist in range(record_length):
                        if top1_dist <= dist + 1:
                            top1_count[dist] += 1
                        if top2_dist <= dist + 1:
                            top2_count[dist] += 1
                        if top3_dist <= dist + 1:
                            top3_count[dist] += 1
                        if top5_dist <= dist + 1:
                            top5_count[dist] += 1
                        if top10_dist <= dist + 1:
                            top10_count[dist] += 1

                    top1_dists += top1_dist
                    top2_dists += top2_dist
                    top3_dists += top3_dist
                    top5_dists += top5_dist
                    top10_dists += top10_dist

                    if i % 4 == 0:
                        print(f'{iters} iter {int(i/4)}th data: gt: {label[i]}, top3: {top_class[i][:3].cpu().numpy()}, top1 dist: {top1_dist:.3f}, top2 dist: {top2_dist:.3f}, top3 dist: {top3_dist:.3f}')

                top1_accuracy_all += sum(top1) * label.shape[0] / len(top1)
                top2_accuracy_all += sum(top2) * label.shape[0] / len(top2)
                top3_accuracy_all += sum(top3) * label.shape[0] / len(top3)
                top5_accuracy_all += sum(top5) * label.shape[0] / len(top5)
                top10_accuracy_all += sum(top10) * label.shape[0] / len(top10)
            
            loss = criterion(output, label)
            loss_all += loss.item()
            
            # print(iters, '/', len(loader), ', top1:', sum(top1) / len(top1), '')
            if eval_dist:
                print(f'{iters}/{len(loader)}, top1: {sum(top1) / len(top1):.3f}, top2: {sum(top2) / len(top2):.3f}, top3: {sum(top3) / len(top3):.3f}, top5: {sum(top5) / len(top5):.3f}, top10: {sum(top10) / len(top10):.3f}')

    print(f'[Warning] {nan_count} Nan result found in total.')
    
    if eval_dist:
        top1_count = [i / len_of_dataset for i in top1_count]
        top2_count = [i / len_of_dataset for i in top2_count]
        top3_count = [i / len_of_dataset for i in top3_count]
        top5_count = [i / len_of_dataset for i in top5_count]
        top10_count = [i / len_of_dataset for i in top10_count]
        
        return loss_all / len_of_dataset, top1_accuracy_all / len_of_dataset, top2_accuracy_all / len_of_dataset, top3_accuracy_all / len_of_dataset, top5_accuracy_all / len_of_dataset, top10_accuracy_all / len_of_dataset, top1_dists / len_of_dataset, top2_dists / len_of_dataset, top3_dists / len_of_dataset, top5_dists / len_of_dataset, top10_dists / len_of_dataset, top1_count, top2_count, top3_count, top5_count, top10_count
    else:
        return loss_all / len_of_dataset, accuracy_all / len_of_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', dest='ckpt', type=str, default='', required=True, help='the ckpt to be evaluated')
    parser.add_argument('--dataset', dest='dataset', type=str, default='icube', choices=['icube', 'wcp'], help='dataset to be evaluated on')
    parser.add_argument('--network', dest='network', type=str, default='gcn', choices=['gcn', 'gat'], help='network to be evaluated on')
    args = parser.parse_args()


    if args.dataset == 'icube':
        path = osp.join(osp.abspath(''), 'data', 'icube')
    else:
        path = osp.join(osp.abspath(''), 'data', 'wcp')

    test_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    if args.dataset == 'icube':
        val_dataset = IndoorDataset(data_dir=osp.join(path, 'icube_test_rearange'), transform=test_transform)
    else:
        val_dataset = IndoorDataset(data_dir=osp.join(path, 'test_rearange_20'), transform=test_transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    print('len of validation set:', len(val_dataset), ', len of validation loader:', len(val_loader))

    # Load checkpoints and evaluate
    model = LocationNet(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()

    # Load checkpoints and evaluate
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])

    val_loss, top1_acc_val, top2_acc_val, top3_acc_val, top5_acc_val, top10_acc_val, top1_dists_val, top2_dists_val, top3_dists_val, top5_dists_val, top10_dists_val, top1_count_val, top2_count_val, top3_count_val, top5_count_val, top10_count_val = evaluate(val_loader, len(val_dataset)*4, args, eval_dist=True)
    print('[Final Result] Top1 Acc: {:.5f}, Top2 Acc: {:.5f}, Top3 Acc: {:.5f}, Top5 Acc: {:.5f}, Top10 Acc: {:.5f}, Test Loss: {:.5f}'.format(
        top1_acc_val, top2_acc_val, top3_acc_val, top5_acc_val, top10_acc_val, val_loss))
    print('[Final Result] Top1 dist: {:.5f}, Top2 dist: {:.5f}, Top3 dist: {:.5f}, Top5 dist: {:.5f}, Top10 dist: {:.5f}'.format(
        top1_dists_val, top2_dists_val, top3_dists_val, top5_dists_val, top10_dists_val))
    print('top1_count:', top1_count_val)
    # print('top2_count:', top2_count_val)
    # print('top3_count:', top3_count_val)
    # print('top5_count:', top5_count_val)
    # print('top10_count:', top10_count_val)

    # val acc: 
    np.save(args.ckpt + '-top1_count.npy', top1_count_val)
