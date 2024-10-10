import json
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision import datasets as dset
import torchvision

from .aptos import APTOS2019

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, test_transform

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = torchvision.datasets.ImageFolder(path + '/train', train_transform)
    np.random.seed(seed)
    
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(7))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_aptos_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019(path, train=True, transforms = train_transform)

    np.random.seed(seed)
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(5))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader


def get_mnist_noise_dataset(dataname, noise_rate = 0.2, batch_size = 32, seed = 0):
    # from medmnist import NoduleMNIST3D
    from medmnist import PathMNIST, BloodMNIST, OCTMNIST, TissueMNIST, OrganCMNIST
    train_transform, test_transform = get_transform()

    if dataname == 'pathmnist':
        train_data = PathMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = PathMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 9
    if dataname == 'bloodmnist':
        train_data = BloodMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = BloodMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'octmnist':
        train_data = OCTMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OCTMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 4
    if dataname == 'tissuemnist':
        train_data = TissueMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = TissueMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'organcmnist':
        train_data = OrganCMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OrganCMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 11

    np.random.seed(seed)
    # new_imgs = []
    new_labels =[]
    for i in range(len(train_data.imgs)):
        if np.random.rand() > noise_rate: # clean sample:
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(train_data.labels[i][0])
        else:
            label_index = list(range(num_classes))
            label_index.remove(train_data.labels[i])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(new_label)
    # train_data.imgs = new_imgs
    train_data.labels = new_labels

    new_labels = []
    for i in range(len(test_data.labels)):
        new_labels.append(test_data.labels[i][0])
    test_data.labels = new_labels

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader