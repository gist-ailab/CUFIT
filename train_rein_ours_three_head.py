import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein

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
        
    num_samples = {}
    for i in range(config['num_classes']):
        num_samples[i] = 0
    for sample in train_loader.dataset:
        num_samples[sample[1]]+=1
    print(num_samples)
    
    class_weight = torch.tensor([sum(num_samples.values())/num_samples[x] for x in num_samples])
    print(class_weight)
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant
    # model = timm.create_model(network, pretrained=True, num_classes=2) 
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    # print(model.state_dict()['blocks.11.mlp.fc2.weight'])
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.to(device)

    model.eval()
    model2.eval()

    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)
    saver = timm.utils.CheckpointSaver(model2, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    avg_accuracy = 0.0
    avg_kappa = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        model2.train()
        total_loss = 0
        total = 0
        correct = 0
        correct2 = 0
        correct_linear = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)

            features_rein2 = model2.forward_features(inputs)
            features_rein2 = features_rein2[:, 0, :]
            outputs2 = model2.linear_rein(features_rein2)

            with torch.no_grad():
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)
            # print(outputs.shape, outputs_.shape)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets)

                pred2 = outputs.max(1).indices
                linear_accurate2 = (pred2==targets)

            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_rein2 = linear_accurate2*criterion(outputs2, targets)
            loss_linear = criterion(outputs_, targets)
            loss = loss_linear.mean()+loss_rein.mean()#+ loss_rein2.mean()
            loss.backward()            
            optimizer.step() # + outputs_

            optimizer2.zero_grad()
            loss_rein2.mean().backward()
            optimizer2.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted = outputs2[:len(targets)].max(1)            
            correct2 += predicted.eq(targets).sum().item()   

            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc2: %.3f%% | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct2/total, 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        model2.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model2, valid_loader, device)
        valid_accuracy_ = utils.validation_accuracy(model, valid_loader, device)
        valid_accuracy_linear = utils.validation_accuracy(model, valid_loader, device, mode='no_rein')
        
        scheduler.step()
        scheduler2.step()
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
if __name__ =='__main__':
    train()