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
from torch.utils.data import Subset, DataLoader, random_split
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import copy
import json
import synthetic_folder

cudnn.benchmark = True
plt.ion()   # interactive mode

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

def save_test_results(filename, data):
    """
        Saves a confusion matrix to a csv file
    """
    data = data.detach().cpu().clone().numpy()
    path = filename + 'eval.csv'
    pd.DataFrame(data).to_csv(path)

def save_train_history(filename, data):
    """
        Saves an array of train/val history to a csv file of the given name
    """
    data = data.detach().cpu().clone().numpy()
    path = filename + '.csv'
    np.savetext(path, data, delimiter=',')

# downloads food101 or loads it if already downloaded, creates a dataset with randomcrop, horizontalflip, tensor, and normalize transformations
# param1: True if want to redownload, false otherwise
def get_food101(root = "/home/dzhang/home/dzhang/efficientnet/data", download=False):
    dataset = torchvision.datasets.Food101(root=root, download=download, transform=data_transforms['train'])
    return dataset 

def load_synthetic(root):
    # synthetic = torchvision.datasets.ImageFolder(root=root, transform=data_transforms, class_to_idx=class_to_idx_dict)
    synthetic = synthetic_folder.SyntheticFolder(root=root, transform=data_transforms['train'])
    return synthetic

# takes a dataset and randomly trims it into a dataset with n samples
# param1: 
def random_trim(dataset, n):
    """
        Returns a trimmed dataset randomly to the specified length.

        Split behavior defined by torch.utils.data.random_split().
        
        Args:
            dataset (Dataset): dataset to be split
            n (int): number of samples in trimmed dataset  

    """
    assert n < len(dataset)
    prop = n/len(dataset)
    split = random_split(dataset=dataset, lengths=[prop, 1-prop])
    return split[0]

# counts the class distribution for classes in a subset
# param1: original dataset, param2: subset of interest
def get_class_distribution(dataset, subset):
    train_classes = [dataset.__getitem__(i)[1] for i in subset.indices]
    print(Counter(train_classes)) # if doesn' work: Counter(i.item() for i in train_classes)

# counts the class distribution for classes in a dataset
# param1: original dataset
def get_class_distribution(dataloader):
    train_classes = []
    for step, (x,y) in enumerate(dataloader):
        train_classes += y.detach().tolist()
    print(Counter(train_classes))



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
# post: a dictionary with train, val, test, and discarded idx
def get_shortened_idx(idx_dict, shortened_class, val_split_regular, val_split_minority, amount_to_discard, test_size=.2):
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
def load_idx_dict(dataset, n, json='idx.json', retrace=False):
    if retrace:
        return get_idx(dataset, n)
    else:
        print("loading idx from " + json)
        idx_file = open(json, "r")
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
def get_dataloader_shapes_distribution(dataset, datasets, dataloaders):
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
    print('train dist')
    get_class_distribution(dataset, datasets['train'])
    print('val dist')
    get_class_distribution(dataset, datasets['val'])
    print('test dist')
    get_class_distribution(dataset, datasets['test'])

# just the dataloader
def get_dataloader_shapes_distribution(dataloaders):
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
    get_class_distribution(dataloaders['train'])
    get_class_distribution(dataloaders['val'])
    get_class_distribution(dataloaders['test'])

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

def make_criterion_optimizer(model, learning_rate=1e-4, amsgrad=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
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
def add_synthetic(root, dataset):
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
    synthetic = torchvision.datasets.ImageFolder(root=root, transform=data_transforms)
    newDataset = torch.utils.data.ConcatDataset(datasets=[dataset, synthetic])
    return newDataset


# loads a idx dict
def load_used_idxs(root):
    print("loading idx from " + root)
    idx_file = open(root, "r")
    json_dict = idx_file.read()
    idx_dict = json.loads(json_dict)
    idx_file.close()
    return idx_dict



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
              
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def eval_model(model, dataloaders, nb_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                # display images or save sample idx that model incorrectly predicts here
                confusion_matrix[t.long(), p.long()] += 1
    print('Confusion matrix: ')
    print(confusion_matrix)
    print('Accuracy by class: ')
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    return confusion_matrix
    

    