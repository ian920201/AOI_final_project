import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def clear_csv(file_path):
    f = open(file_path, "w")
    f.truncate(0)
    f.close()

def write_csv(file_path, epoch, acc):
    f = open(file_path, "a")
    f.write(f"{epoch},{acc}\n")
    f.close()

def has_converged(accuracy_list, threshold = 0.01, patience = 5):
    if (len(accuracy_list) < patience + 1):
        return False
    for i in range (1, patience + 1):
        if (abs(accuracy_list[-i] - accuracy_list[-(i + 1)]) > threshold):
            return False
    return True

def train_data(filedir):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(filedir, transform=transform_train)
    return dataset

def test_data(filedir):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(filedir, transform=transform_test)
    return dataset