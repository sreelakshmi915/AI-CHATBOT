import json
from main import tokenization, stem, bow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Neuralnetwork


import numpy as np
with open('data.json', 'r') as f:
    data = json.load(f)
full_words = []
tags = []
xy = []
for dat in data['chat']:
    tag = dat['tag']
    tags.append(tag)
    for pattern in dat['patterns']:
        t = tokenization(pattern)
        full_words.extend(t)
        xy.append((t, tag))
word_ignore = ['?', '!', '.', ',']
full_words = [stem(t) for t in full_words if t not in word_ignore]

full_words = sorted(set(full_words))
tags = sorted(set(tags))
print(full_words)
print(tags)
XX_train = []
yY_train = []
for (patt_tag, taged) in xy:
    bag = bow(patt_tag, full_words)
    XX_train.append(bag)
    label = tags.index(taged)
    yY_train.append(label) #  cross entropy loss

    X_train = np.array(XX_train)
    y_train = np.array(yY_train)
#creating pytorch dataset
class Chat_data(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) #length of x_train
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        #dataset[idx]
    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size =8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000
print(input_size, output_size)
dataset = Chat_data()
train_loader = DataLoader(dataset= dataset, batch_size=batch_size, shuffle=True, num_workers=0) # creating dataloder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Neuralnetwork(input_size,hidden_size,output_size).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for(words,labels)in train_loader:
        words = words.to(device)
        labels = labels.to(device)


#forward pass
        outputs = model(words)
        loss = criterion(outputs,labels)
        #back propagation and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch+1)%100 ==0:
        print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss , loass ={loss.item():.4f}')





