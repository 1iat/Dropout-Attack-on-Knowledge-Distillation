import math
import torch
import torchvision
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/jupyter-iec_roadquality/Security/1iat/DropoutAttack/modules/custom-datasets')
#from mnist_activations_dataset import MNIST_ACTIVATIONS_DATASET
#from slicing_dataloader import FastTensorDataLoader

def load_mnist(batch_size, transform):
    """
    Load the MNIST Dataset

        Parameters:
            batch_size: The batch sizes of the returned data loaders

        Return:
            trainset: The MNIST train set
            validationset: The MNIST validation set
            testset: The MNIST test set
            trainloader: The MNIST trainset data loader
            validationloader: The MNIST validationset data loader 
            testloader: The MNIST testset data loader
    """
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    # 90/10
    trainset, validationset = torch.utils.data.random_split(trainset, [54000, 6000])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainset, validationset, testset, trainloader, validationloader, testloader


def load_cifar(batch_size, transform):
    """
    Load the CIFAR-10 Dataset

        Parameters:
            batch_size: The batch sizes of the returned data loaders
            transform: the transform to apply

        Return:
            trainset: The CIFAR train set
            validationset: The CIFAR validation set
            testset: The CIFAR test set
            trainloader: The CIFAR trainset data loader
            validationloader: The CIFAR validationset data loader 
            testloader: The CIFAR testset data loader
    """
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # 90/10
    trainset, validationset = torch.utils.data.random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainset, validationset, testset, trainloader, validationloader, testloader

def load_cifar100(batch_size, transform):
    """
    Load the CIFAR-100 Dataset

        Parameters:
            batch_size: The batch sizes of the returned data loaders
            transform: the transform to apply

        Return:
            trainset: The CIFAR train set
            validationset: The CIFAR validation set
            testset: The CIFAR test set
            trainloader: The CIFAR trainset data loader
            validationloader: The CIFAR validationset data loader 
            testloader: The CIFAR testset data loader
    """
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    # 90/10
    trainset, validationset = torch.utils.data.random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainset, validationset, testset, trainloader, validationloader, testloader

def surrogate_actual_split(trainset, validationset, testset, batch_size):
    """
    Splits a dataset into a dataset for a surrogate and actual model

        Parameters:
            trainset: the train set
            validationset: the validation set
            testset: the test set
        
        Returns:
            A tuple of tuples of size 4. The following is what is in each position:
                0: (surrogate train set, surrogate validation set, surrogate test set)
                1: (actual train set, actual validation set, actual test set)
                2: (surrogate train dataloader, surrogate validation dataloader, surrogate test dataloader)
                3: (actual train dataloader, actual validation dataloader, surrogate test dataloader)

    """
    surrogate_size = math.floor(0.5 * len(trainset))
    trainset_surrogate, trainset_actual = torch.utils.data.random_split(
        trainset, [surrogate_size, len(trainset) - surrogate_size]
    )
    surrogate_size = math.floor(0.5 * len(validationset))
    validset_surrogate, validset_actual = torch.utils.data.random_split(
        validationset, [surrogate_size, len(validationset) - surrogate_size]
    )
    surrogate_size = math.floor(0.5 * len(testset))
    testset_surrogate, testset_actual = torch.utils.data.random_split(
        testset, [surrogate_size, len(testset) - surrogate_size]
    )
    trainloader_surrogate = torch.utils.data.DataLoader(
        trainset_surrogate, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validloader_surrogate = torch.utils.data.DataLoader(
        validset_surrogate, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader_surrogate = torch.utils.data.DataLoader(
        testset_surrogate, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainloader_actual = torch.utils.data.DataLoader(
        trainset_actual, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validloader_actual = torch.utils.data.DataLoader(
        validset_actual, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader_actual = torch.utils.data.DataLoader(
        testset_actual, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return (
        (trainset_surrogate, validset_surrogate, testset_surrogate),
        (trainset_actual, validset_actual, testset_actual),
        (trainloader_surrogate, validloader_surrogate, testloader_surrogate),
        (trainloader_actual, validloader_actual, testloader_actual),
    )