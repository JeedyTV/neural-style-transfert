#inspired by HW2.
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ourvgg import VGG
from torchvision import datasets, transforms, utils
from bot import send_message


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

learning_rate = 0.001
num_epochs = 30
batch_size = 128

#load our CNN 
network = VGG()

#set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay = 0.0005, momentum = 0.9) 


#Define the transform to apply to the CIFAR-100 dataset

#compute the mean and std of cifar 100
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load the CIFAR-100 training dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform)

# Get the pixel values of all training images
train_data = train_dataset.data

# Calculate the mean and standard deviation
mean = np.mean(train_data, axis=(0, 1, 2)) / 255.0  # Normalize by 255.0 to obtain values in the range [0, 1]
std = np.std(train_data, axis=(0, 1, 2)) / 255.0

normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

transform_train = transforms.Compose([
                           transforms.Resize((224,224)),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(224, padding=10),
                           transforms.ToTensor(),
                           normalize,
                       ])
transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])

trainset = datasets.CIFAR100(root='./data', train=True, transform=transform_train)
testset = datasets.CIFAR100(root='./data', train=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

def train(num_epochs):
    train_avg_loss = []
    test_avg_loss = []
    test_accuracy = []

    network.to(device)
    network.train()
    for i in tqdm(range(num_epochs)):
        print(f"start epoch {i}")
        train_losses = []
        test_losses = []
        for x, y in tqdm(trainloader):
            x = x.to(device)
            y = y.to(device)

            pred = network(x)
            loss = criterion(pred, y)
            train_losses.append(loss.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        network.eval()
        with torch.no_grad():   
            correct = 0
            
            for x, y in tqdm(testloader):
                
                x = x.to(device)
                y = y.to(device)

                pred = network(x)
                loss = criterion(pred, y)
                test_losses.append(loss)
                
                y_pred = pred.argmax(dim=-1)
                correct = correct + (y_pred == y).sum()

            accuracy = correct / len(testset)
            test_accuracy.append(accuracy)
            t = torch.mean(torch.stack(train_losses))
            t1 = torch.mean(torch.stack(test_losses))
            train_avg_loss.append(t)
            test_avg_loss.append(t1)
            m = f"epoch {i} train_loss: {t.item()} ,test_loss: {t1.item()} ,accuracy: {accuracy.item()}"
            print(m)
            send_message(m)
   
    network.to("cpu")
    return train_avg_loss, test_avg_loss, test_accuracy

torch.cuda.empty_cache()

train_avg_loss, test_avg_loss, test_accuracy = train(num_epochs)

train_avg_loss = [loss.cpu() for loss in train_avg_loss]
test_avg_loss = [loss.cpu() for loss in test_avg_loss]
test_accuracy = [acc.cpu() for acc in test_accuracy]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(test_accuracy , label='test accuracy')
ax1.legend()
ax2.plot(train_avg_loss, label='train loss')
ax2.plot(test_avg_loss, label='test loss')
ax2.legend()

plt.savefig(f'model_{learning_rate}.png')

torch.save(network.state_dict(), f'model_{learning_rate}.pth')