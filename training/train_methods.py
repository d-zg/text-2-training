from __future__ import print_function, division
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import json
cudnn.benchmark = True
plt.ion()   # interactive mode

# downloads food101 or loads it if already downloaded, creates a dataset with randomcrop, horizontalflip, tensor, and normalize transformations
# param1: True if want to redownload, false otherwise
def get_food101(root = "/home/dzhang/home/dzhang/efficientnet/data", download=False):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(438),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dataset = torchvision.datasets.Food101(root=root, download=download, transform=data_transforms['train'])
    return dataset 

# counts the class distribution for classes in a subset
# param1: original dataset, param2: subset of interest
def get_class_distribution(dataset, subset):
    train_classes = [dataset.__getitem__(i)[1] for i in subset.indices]
    print(Counter(train_classes)) # if doesn' work: Counter(i.item() for i in train_classes)

# creates a subset with the first num_classes+1 classes
# param1: original dataset, param2: the max class number we want to include
def make_subset(dataset, num_classes):
    idx = [i for i in range(len(dataset)) if dataset.__getitem__(i)[1] in range(num_classes)]
    return Subset(dataset, idx)


# creates a dict with all image_idx categorized into class id
# param1: original dataset. param2: the max class id we want to sort/include in our dict
# post: dict with keys as class_ids we care about as specified by n, values of an array with all image_idx from the class_id key
def get_idx(dataset, n):
    idx_dict = {}
    for i in range(len(dataset)):
        label = dataset.__getitem__(i)[1]
        if label in range(n): # can probably replace with <=
            if label in idx_dict:
                idx_dict[label].append(i)
            else:
                idx_dict[label] = [i]
        if i % 2000 == 0:
            print(i)
    return idx_dict

# unnecessary
# def get_idx_dict(dataset, n, load=True):
#     if not load: 
#         return load_idx_dict(dataset, 6)
#     else: 
#         return get_idx(dataset, n)

# returns the label name for a class idx in the dataset
# param1: dataset of interest. param2: class_idx we care about 
def get_label_name(dataset, i):
    class_swap = {v:k for k, v in dataset.class_to_idx.items()}
    return class_swap[i]

# Makes a train, val, test split with idxs, simultaneously throws out idxs from the minority class
# param1: the original idx_dict. param2: idx of the class we want to limit. param3: val split proportion of non-shortened classes. 
# param4: val split of the shortened class. param5: number of minority class samples to discard. param6: train/test split proportion. 
# this is all in one because want to have more control over discarding-- making sure we have same number of test points for minority class
def get_shortened_idx(idx_dict, shortened_class, val_split_regular, val_split_minority, amount_to_discard, test_size=.2):
    new_dict = {}
    train_idx = []
    val_idx = []
    test_idx = []
    for k, v in idx_dict.items():
        if k != shortened_class:
            class_i_train, class_i_test = train_test_split(v, test_size=test_size)
            class_i_train, class_i_val = train_test_split(class_i_train, test_size=val_split_regular)
            train_idx += class_i_train
            val_idx += class_i_val
            test_idx += class_i_test
    shortened_list = idx_dict[shortened_class][amount_to_discard:] # replace with random sampling and tracking the discarded indices later. Just throws away first n samples for now
    minority_train, minority_val = train_test_split(shortened_list, test_size=val_split_minority)
    train_idx += minority_train
    val_idx += minority_val
    discarded_idx = idx_dict[shortened_class][:amount_to_discard] 
    test_idx += discarded_idx[:int(round(750*test_size))]
    
    split_idx = {}
    split_idx['train'] = train_idx
    split_idx['val'] = val_idx
    split_idx['discarded'] = idx_dict[shortened_class][:amount_to_discard] # the discarded ids
    split_idx['test'] = test_idx # 
    return split_idx

# splits into trian, val, test subsets and returns them as a datasets dict
# param1: dataset. param2: idx_dict with the train, val, test idx 
def train_val_split_idx(dataset, idx):
    datasets = {}
    datasets['train'] = Subset(dataset, idx['train'])
    datasets['val'] = Subset(dataset, idx['val'])
    datasets['test'] = Subset(dataset, idx['test'])
    return datasets


# makes dataloaders for datasets 
# param1: a dict with train, val, test datasets. param2: batch size of our loaders. 
# post: a dict with train, val, and test dataloaders. 
def get_dataloaders(datasets, batch_size):
    train_dataloader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)

    val_dataloader = DataLoader(datasets['val'], batch_size=batch_size, shuffle =True)

    test_dataloader = DataLoader(datasets['test'], batch_size=batch_size, shuffle =True)
    dataloaders = {}

    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    dataloaders['test'] = test_dataloader
    return dataloaders

# actually no idea lol 
def keystoint(x):
    return {int(k): v for k, v in x.items()}

# loads the idx_dict a dataset
# param1: dataset. param2: classes of interest. param3: whether or not to load previous idx_dict 
def load_idx_dict(dataset, n, retrace=False):
    if retrace:
        return get_idx(dataset, n)
    else:
        print("loading idx from idx.json")
        idx_file = open("idx.json", "r")
        json_dict = idx_file.read()
        idx_dict = json.loads(json_dict, object_hook=keystoint)
        idx_file.close()
        return idx_dict

# given a dataset, splits dataset into a dictionary with train, val, and test subsets
# param1: original dataset, param2: train/val split (proportion of train data as val), param3: train/test split size (proportion of all data as test)
def train_val_full_dataset(dataset, val_split = .25, test_size = .2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_size)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

# prints a counter of the number of images in each class and displays the next image in the loader
# param1: dataloders, param2: the full dataset, param3: the subset in question
def get_dataloader_shapes_distribution(dataloaders, datasets, dataset):
    train_features, train_labels = next(iter(dataloaders['train']))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # print(f"Feature batch: {train_features}")
    # print(f"Labels batch: {train_labels}")
    img = train_features[0].squeeze()
    img = img[0, :, :]
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print(train_features.shape)
    get_class_distribution(dataset, datasets['train'])
    get_class_distribution(dataset, datasets['val'])
    get_class_distribution(dataset, datasets['test'])

# makes an identity layer
# used to functionally delete layers when replacing
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

# fits an efficientnet model
def fit_efficientnet_shape(n):
    # fitting efficientnet shape 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.efficientnet_v2_s(pretrained=False, weights=None)
  
    # removing avg pool layer 
    model.avgpool = Identity()
    model.classifier = nn.Linear(184320,n)

    model.to(device)
    return model

def make_criterion_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return criterion, optimizer

# saving and loading model
def save_checkpoint(state, filename="efficientnet_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

# load model
def load_checkpoint(model, optimizer, filename="/home/dzhang/home/dzhang/efficientnet/efficientnet_checkpoint.pth.tar"):
    print("loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# adds synthetic data in imagefolder format to an existing dataset
# param1: root of the folder of images we want to add. param2: original dataset. transform: transform we want to put on new imagefolder.
def add_synthetic(root, dataset, transform):
    synthetic = torchvision.datasets.ImageFolder(root=root, transform=transform)
    newDataset = torch.utils.data.ConcatDataset(dataset, synthetic)
    return newDataset


