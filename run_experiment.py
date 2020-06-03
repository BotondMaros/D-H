import modified_mobilenet_1k_baseline as base #import scripts I wrote
import torch.nn as nn
import torch.optim as optim
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

PATH = "data"
data, labels = base.load_all(PATH)
data = base.get_shape_32x32(data)

indeces = base.get_100_from_each(labels)

train_data, train_labels, test_data, test_labels = base.normalize_split(indeces, data, labels)
train_dataloader, test_dataloader = base.create_dataset(train_data, test_data, train_labels, test_labels, 32)

net = base.mobilenet_v2(pretrained=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=4e-5)

saved_model = 'modified_model_1k_2'
checkpoint = torch.load(saved_model,map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
criterion = checkpoint['loss']

correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data

        outputs,outputs2 = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

print('Accuracy of the network on the 59000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data

        outputs,outputs2 = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))