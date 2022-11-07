#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import pathlib

pathlib.Path().resolve()


# In[13]:


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

dataset = torchvision.datasets.Food101(root="/home/dzhang/home/dzhang/efficientnet/data", download=False, transform=data_transforms['train'])


# In[14]:



def get_class_distribution(dataset, subset):
    train_classes = [dataset.__getitem__(i)[1] for i in subset.indices]
    print(Counter(train_classes)) # if doesn' work: Counter(i.item() for i in train_classes)
    
def make_subset(dataset, num_classes):
    idx = [i for i in range(len(dataset)) if dataset.__getitem__(i)[1] in range(num_classes)]
    return Subset(dataset, idx)

def get_idx(dataset, n):
    idx_dict = {}
    for i in range(len(dataset)):
        label = dataset.__getitem__(i)[1]
        if label in range(n):
            if label in idx_dict:
                idx_dict[label].append(i)
            else:
                idx_dict[label] = [i]
        if i % 2000 == 0:
            print(i)
    return idx_dict
    
def get_label_name(dataset, i):
    class_swap = {v:k for k, v in dataset.class_to_idx.items()}
    return class_swap[i]
    
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

def train_val_split_idx(dataset, idx):
    datasets = {}
    datasets['train'] = Subset(dataset, idx['train'])
    datasets['val'] = Subset(dataset, idx['val'])
    datasets['test'] = Subset(dataset, idx['test'])
    return datasets

def get_dataloaders(datasets, batch_size):
    train_dataloader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)

    val_dataloader = DataLoader(datasets['val'], batch_size=batch_size, shuffle =True)

    test_dataloader = DataLoader(datasets['test'], batch_size=batch_size, shuffle =True)
    dataloaders = {}

    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    dataloaders['test'] = test_dataloader
    return dataloaders

def keystoint(x):
    return {int(k): v for k, v in x.items()}

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
        
def train_val_full_dataset(dataset, val_split = .25, test_size = .2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_size)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets


# In[15]:


# idx_file = open("idx.json", "w")
# json.dump(idx_dict, idx_file)

# def keystoint(x):
#     return {int(k): v for k, v in x.items()}

# idx_file = open("idx.json", "r")
# json_dict = idx_file.read()
# idx_dict = json.loads(json_dict, object_hook=keystoint)
# idx_file.close()

idx_dict = load_idx_dict(dataset, 6)


# In[16]:


train_val_idx = get_shortened_idx(idx_dict, 5, .25, .25, 625) # making minority class (class 5) with 625 thrownaway points of data as training. 125 images vs 600. Both have .75 train validation split. All thrownaway points are put in test 

datasets = train_val_split_idx(dataset, train_val_idx) # making subsets from idx above


# In[17]:


train_val_idx = get_shortened_idx(idx_dict, 5, .25, .25, 625) # making minority class (class 5) with 625 thrownaway points of data as training. 125 images vs 600. Both have .75 train validation split. All thrownaway points are put in test 

datasets = train_val_split_idx(dataset, train_val_idx) # making subsets from idx above

batch_size = 32

dataloaders = get_dataloaders(datasets=datasets, batch_size=batch_size)


def get_dataloader_shapes_distribution(dataloaders, datasets):
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

get_dataloader_shapes_distribution(dataloaders, datasets)


# In[54]:


# for loading and training on full untrimmed set

full_datasets = train_val_full_dataset(dataset)

full_dataloaders = get_dataloaders(datasets=full_datasets, batch_size=batch_size)

# get_dataloader_shapes_distribution(full_dataloaders, full_datasets)


# In[19]:


class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

def fit_efficientnet_shape(n):
    # fitting efficientnet shape 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.efficientnet_v2_s(pretrained=False, weights=None)
    # print(model)

  
    # removing avg pool layer 
    model.avgpool = Identity()
    # model.features.avgpool = Identity()
    model.classifier = nn.Linear(184320,n)

    model.to(device)
    return model
    # print(model)


# In[ ]:


# fitting efficientnet shape 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.efficientnet_v2_s(pretrained=False, weights=None)
# print(model)

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  
  def forward(self, x):
    return x

# removing avg pool layer 
model.avgpool = Identity()
# model.features.avgpool = Identity()
model.classifier = nn.Linear(184320,6)

model.to(device)
# print(model)


# In[21]:


# Hyperparameters
num_classes = 6
learning_rate = 2e-6
num_epochs = 25
load_model = False
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def make_criterion_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return criterion, optimizer

# saving and loading model
def save_checkpoint(state, filename="efficientnet_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename="/home/dzhang/home/dzhang/efficientnet/efficientnet_checkpoint.pth.tar"):
    print("loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
# # train_features = train_features.to(device)
# outputs = model(train_features)
# # print(preds.shape)
# print(train_labels.shape)

# x, preds = torch.max(outputs, 1)
# print(preds.shape)
# print(preds)
# print(train_labels)


# In[22]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_criterion_optimizer(data, load=True):
    if data == 'trimmed':
        model = fit_efficientnet_shape(6)
        criterion, optimizer = make_criterion_optimizer(model)
        if load:
            load_checkpoint(model=model, optimizer=optimizer)
        return model, criterion, optimizer    
    

model, criterion, optimizer = load_model_criterion_optimizer(data='trimmed')

# load_checkpoint()


# In[23]:


def change_lr(optimizer, learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


# In[24]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

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
    

    


# In[25]:


# change_lr(optimizer, 1e-8)

train_val_idx = get_shortened_idx(idx_dict, 5, .25, .25, 625) # making minority class (class 5) with 625 thrownaway points of data as training. 125 images vs 600. Both have .75 train validation split. All thrownaway points are put in test 

datasets = train_val_split_idx(dataset, train_val_idx) # making subsets from idx above

batch_size = 32

dataloaders = get_dataloaders(datasets=datasets, batch_size=batch_size)

model, criterion, optimizer = load_model_criterion_optimizer(data='trimmed')

load_checkpoint(model, optimizer)


# best_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=100) # returns model weights that achieve best validation accuracy 


# In[27]:


# eval_model(model, dataloaders, num_classes)

eval_model(model, dataloaders, num_classes)


# In[96]:



checkpoint = {'state_dict' : best_model[0].state_dict(), 'optimizer' : optimizer.state_dict()}
save_checkpoint(state=checkpoint)


# 

# In[71]:


# for loading and training on full untrimmed set

full_datasets = train_val_full_dataset(dataset)

full_dataloaders = get_dataloaders(datasets=full_datasets, batch_size=batch_size)

get_dataloader_shapes_distribution(full_dataloaders, full_datasets)

fullmodel = fit_efficientnet_shape(101)
fullCriterion, fullOptimizer = make_criterion_optimizer(fullmodel)

        
change_lr(fullOptimizer, 2e-6)

best_full_model = train_model(fullmodel, full_dataloaders, fullCriterion, fullOptimizer, num_epochs=100)


# In[72]:


eval_model(fullmodel, full_dataloaders, 101)

checkpoint = {'state_dict' : best_full_model[0].state_dict(), 'optimizer' : fullOptimizer.state_dict()}
save_checkpoint(state=checkpoint, filename="full_food101efficientnet_checkpoint.pth.tar")


# In[ ]:


few_class_dataset = train_val(dataset)

full_dataloaders = get_dataloaders(datasets=full_datasets, batch_size=batch_size)

get_dataloader_shapes_distribution(full_dataloaders, full_datasets)

fullmodel = fit_efficientnet_shape(101)
fullCriterion, fullOptimizer = make_criterion_optimizer(fullmodel)

        
# change_lr(fullOptimizer, 2e-6)

best_full_model = train_model(fullmodel, full_dataloaders, fullCriterion, fullOptimizer, num_epochs=100)

