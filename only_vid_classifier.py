#Author: Syrine HADDAD

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from torchview import draw_graph
import graphviz


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*30*62, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        self.avgpool = nn.AdaptiveAvgPool2d((30, 62))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        #x = x.view(-1, 128*30*62)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(data_loader, criterion, net, device, optimizer):
    train_loss = []
    val_loss = []
    val_accuracy = []
    train_accuracy = []

    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        t_loss = 0
        total = 0
        total = 0
        correct = 0
        for i, data in enumerate(data_loader['vid_train']):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            t_loss += loss.item()
            if i % 2 == 1:    # print every 2 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        train_loss.append(t_loss)
        train_accuracy.append(100*correct/total)
        
        correct = 0
        total = 0
        v_loss = 0
        for i, data in enumerate(data_loader['vid_val']):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss.append(v_loss)
        val_accuracy.append(100*correct/total)
        print('Accuracy of the network on the validation dataset: %d %%' % (
                100 * correct / total))

    np.save('train_accuracy_only_video', train_accuracy)
    np.save('train_loss_only_video', train_loss)
    np.save('test_accuracy_only_video', val_accuracy)
    np.save('test_loss_only_video', val_loss)

    print('Finished Training')
    plt.plot(train_loss, 'g', label='Training loss')
    plt.plot(val_loss, 'b', label='validation loss')
    plt.title('Training  & validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(train_accuracy, 'g', label='Training Accuracy')
    plt.plot(val_accuracy, 'b', label='validation Accuracy')
    plt.title('Training  & validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def test(data_loader, criterion, net, device, optimizer):
    # print images
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(8, 8)
    with torch.no_grad():
        for data in data_loader['vid_test']:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    torch.save(net.state_dict(), 'onlyVideo1.pth')           
    np.save('confusion_matrix_only_video', confusion_matrix)
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def only_video(folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print (device)
    # Load the data that needs to be analyzed
    data_transform = {
        'vid_train' : tv.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ]),
        'vid_val' : tv.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]),
        'vid_test' : tv.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])

    }

    datasets = ['vid_train','vid_val','vid_test']
    image_data = {}
    for x in datasets:
        image_data[x] = tv.datasets.ImageFolder(folder + '/' + x, transform=data_transform[x])


    data_loader = {}
    for x in datasets:
        data_loader[x] = torch.utils.data.DataLoader(image_data[x], batch_size=24,
                   shuffle=True, num_workers=0)

    
    net = Net()
    print(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    train(data_loader, criterion, net, device, optimizer)
    test(data_loader, criterion, net, device, optimizer)