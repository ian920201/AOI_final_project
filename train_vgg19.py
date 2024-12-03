import sys
import torch
import torch.nn as nn
import torch.optim as optim
from data_process import clear_csv, write_csv, train_data, test_data
from torchvision import models
from tqdm import tqdm

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available! Training on GPU...')

train_dataset = train_data("./images/train")
test_dataset = test_data("./images/test")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 100),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.5),
    torch.nn.Linear(100, 31)
)

if train_on_gpu:
    model.cuda()

entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train():
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = data
        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        out = model(inputs)
        loss = entropy_loss(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            inputs, labels = data
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            out = model(inputs)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
    acc = correct / len(test_dataset)
    return acc

def main():
    clear_csv("./result_data/result_vgg19.csv")
    n = int(sys.argv[1])
    for epoch in range(1, n + 1):
        print("epoch: " + str(epoch))
        train_loss = train()
        acc = test()
        write_csv("./result_data/result_vgg19.csv", epoch, acc)
        print(f"Epoch [{epoch}/{n}] - Loss: {train_loss:.4f} - Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "./pth/face_vgg19.pth")

if __name__ == "__main__":
    main()
