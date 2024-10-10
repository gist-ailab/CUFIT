import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
from sklearn.metrics import f1_score

import dino_variant


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]


    if args.data == 'ham10000':
        train_loader, valid_loader = utils.get_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'aptos':
        train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
    elif 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noise_dataset(args.data, data_path, batch_size = batch_size,  noise_rate=noise_rate)

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    model = torch.hub.load('facebookresearch/dinov2', model_load)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
        
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    if args.data == 'dr':
        num_samples = {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708}
        class_weight = torch.tensor([1-num_samples[x]/sum(num_samples.values()) for x in num_samples]).to(device)
        print(class_weight)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=1e-3, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)


    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                outputs = model(inputs)
            outputs = model.linear(outputs)
            
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            train_accuracy = correct/total

        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = utils.validation_accuracy(model, valid_loader, device, mode='linear')
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()