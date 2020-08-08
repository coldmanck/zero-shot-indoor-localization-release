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

def evaluate(loader, len_of_dataset, args):
    model.eval()
    
    with open('data/' + args.dataset + '/loc_vec.npy', 'rb') as f:
        loc_coord = np.load(f)[1:]
    n_classes = 214 if args.dataset == 'icube' else 394
    if args.network == 'baseline':
        logits = loc_coord[all_classes] # [classes=214, vec_length=2]
    else:
        with open('data/' + args.dataset + '/loc_vec_trained_' + str(n_classes) + '.npy', 'rb') as f:
            logits = np.load(f)[all_classes] # [classes=214 or 394, vec_length=2, 214 or 394]

    logits = torch.from_numpy(logits).float()
    logits = logits / logits.norm(dim=1, keepdim=True)
    logits = logits.to(device)
    
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
        for data, label in loader:
            iters += 1
            data, label = data.to(device), label.view(-1).to(device)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                                       [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long).to(device)
            output = model(data, edge_index)
            # output = output / output.norm(dim=1, keepdim=True)
            new_output = torch.zeros((output.shape[0], n_classes)).to(device)
            for i in range(output.shape[0]):
                new_output[i] = (output[i] * logits).sum(dim=1)
            top_p, top_class = new_output.topk(10)

            top1 = []
            top2 = []
            top3 = []
            top5 = []
            top10 = []
            for i in range(top_class.shape[0]):
                if any(top_class[i] > n_classes-1) or any(top_class[i] < 0):
                    nan_count += 1
                    print(f'Nan result detected! Count: {nan_count}')
                    top_class[i] = torch.randint(0, n_classes, top_class[i].shape).to(device)
                
                top1 += [1] if label[i] == top_class[i][0] else [0]
                top2 += [1] if label[i] in top_class[i][:2] else [0]
                top3 += [1] if label[i] in top_class[i][:3] else [0]
                top5 += [1] if label[i] in top_class[i][:5] else [0]
                top10 += [1] if label[i] in top_class[i][:10] else [0]

                if (all_dict_to_loc[label[i].cpu().numpy().item()] >= n_classes) or (all_dict_to_loc[label[i].cpu().numpy().item()] < 0) or any([all_dict_to_loc[top_class[i][k].cpu().numpy().item()] >= n_classes for k in range(0, 10)]) or any([all_dict_to_loc[top_class[i][k].cpu().numpy().item()] < 0 for k in range(0, 10)]):
                    import pdb; pdb.set_trace()
                top1_dist = np.linalg.norm(loc_coord[all_dict_to_loc[label[i].cpu().numpy().item()]] - loc_coord[all_dict_to_loc[top_class[i][0].cpu().numpy().item()]])
                top2_dist = sum([np.linalg.norm(loc_coord[all_dict_to_loc[label[i].cpu().numpy().item()]] - loc_coord[all_dict_to_loc[top_class[i][k].cpu().numpy().item()]]) for k in range(0, 2)]) / 2
                top3_dist = sum([np.linalg.norm(loc_coord[all_dict_to_loc[label[i].cpu().numpy().item()]] - loc_coord[all_dict_to_loc[top_class[i][k].cpu().numpy().item()]]) for k in range(0, 3)]) / 3
                top5_dist = sum([np.linalg.norm(loc_coord[all_dict_to_loc[label[i].cpu().numpy().item()]] - loc_coord[all_dict_to_loc[top_class[i][k].cpu().numpy().item()]]) for k in range(0, 5)]) / 5
                top10_dist = sum([np.linalg.norm(loc_coord[all_dict_to_loc[label[i].cpu().numpy().item()]] - loc_coord[all_dict_to_loc[top_class[i][k].cpu().numpy().item()]]) for k in range(0, 10)]) / 10
                
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
                    print(f'{iters} iter {int(i/4)}th data: gt: {all_dict_to_loc[label[i].cpu().numpy().item()]}, top3: {[all_dict_to_loc[i] for i in top_class[i][:3].cpu().numpy().tolist()]}, top1 dist: {top1_dist:.3f}, top2 dist: {top2_dist:.3f}, top3 dist: {top3_dist:.3f}')
                
            top1_accuracy_all += sum(top1) * label.shape[0] / len(top1)
            top2_accuracy_all += sum(top2) * label.shape[0] / len(top2)
            top3_accuracy_all += sum(top3) * label.shape[0] / len(top3)
            top5_accuracy_all += sum(top5) * label.shape[0] / len(top5)
            top10_accuracy_all += sum(top10) * label.shape[0] / len(top10)
            
            # loss = criterion(output, label)
            # loss_all += loss.item()
            
            # print(iters, '/', len(loader), ', top1:', sum(top1) / len(top1), '')
            print(f'{iters}/{len(loader)}, top1: {sum(top1) / len(top1):.3f}, top2: {sum(top2) / len(top2):.3f}, top3: {sum(top3) / len(top3):.3f}, top5: {sum(top5) / len(top5):.3f}, top10: {sum(top10) / len(top10):.3f}')
    
    top1_count = [i / len_of_dataset for i in top1_count]
    top2_count = [i / len_of_dataset for i in top2_count]
    top3_count = [i / len_of_dataset for i in top3_count]
    top5_count = [i / len_of_dataset for i in top5_count]
    top10_count = [i / len_of_dataset for i in top10_count]
    
    print(f'[Warning] {nan_count} Nan result found in total.')
    
    return top1_accuracy_all / len_of_dataset, top2_accuracy_all / len_of_dataset, top3_accuracy_all / len_of_dataset, top5_accuracy_all / len_of_dataset, top10_accuracy_all / len_of_dataset, top1_dists / len_of_dataset, top2_dists / len_of_dataset, top3_dists / len_of_dataset, top5_dists / len_of_dataset, top10_dists / len_of_dataset, top1_count, top2_count, top3_count, top5_count, top10_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', dest='ckpt', type=str, default='', required=True, help='the ckpt to be evaluated')
    parser.add_argument('--dataset', dest='dataset', type=str, default='icube', choices=['icube', 'wcp'], help='dataset to be evaluated on')
    parser.add_argument('--network', dest='network', type=str, default='gcn', choices=['baseline', 'gcn', 'gat'], help='network to be evaluated on')
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

    test_dataset = IndoorDataset(osp.join(path, 'test_zl'), dataset=args.dataset, transform=test_transform)
    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print('len of testing set:', len(test_dataset), ', len of testing loader:', len(test_loader))

    with open('data/' + args.dataset + '/all_loc_to_dict.pkl', 'rb') as f:
        all_loc_to_dict = pickle.load(f)
    all_classes = [i - 1 for i in all_loc_to_dict.keys()]
    all_dict_to_loc = {j-1: i-1 for i,j in all_loc_to_dict.items()}

    model = LocationNet(args, zero_shot=True)
    ckpt = torch.load(args.ckpt)['state_dict']
    del ckpt['fc.weight']
    del ckpt['fc.bias']
    model.load_state_dict(ckpt)

    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss()

    print(f'Evaluating on test set...')
    top1_acc, top2_acc, top3_acc, top5_acc, top10_acc, top1_dists, top2_dists, top3_dists, top5_dists, top10_dists, top1_count, top2_count, top3_count, top5_count, top10_count = evaluate(test_loader, len(test_dataset)*4, args)

    print('[Final Result] Top1 Acc: {:.5f}, Top2 Acc: {:.5f}, Top3 Acc: {:.5f}, Top5 Acc: {:.5f}, Top10 Acc: {:.5f}'.format(
        top1_acc, top2_acc, top3_acc, top5_acc, top10_acc))
    print('[Final Result] Top1 dist: {:.5f}, Top2 dist: {:.5f}, Top3 dist: {:.5f}, Top5 dist: {:.5f}, Top10 dist: {:.5f}'.format(
        top1_dists, top2_dists, top3_dists, top5_dists, top10_dists))
    print('top1_count:', top1_count)
    # print('top2_count:', top2_count)
    # print('top3_count:', top3_count)
    # print('top5_count:', top5_count)
    # print('top10_count:', top10_count)

    np.save(args.ckpt + '-top1_count.npy', top1_count)
