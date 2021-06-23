import json
from main import tokenization,stem,bow
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import numpy as np
with open('data.json','r') as f:
    data = json.load(f)
full_words =[]
tags =[]
xy =[]
for dat in data['chat']:
    tag = dat['tag']
    tags.append(tag)
    for pattern in dat['patterns']:
        t =tokenization(pattern)
        full_words.extend(t)
        xy.append((t,tag))
word_ignore = ['?','!','.',',']
full_words = [stem(t) for t in full_words if t not in word_ignore]

full_words =sorted(set(full_words))
tags = sorted(set(tags))
print(full_words)
print(tags)
X_tain = []
y_train =[]
for (patt_tag,taged) in xy:
    bag = bow(patt_tag,full_words)
    X_tain.append(bag)
    label = tags.index(taged)
    y_train.append(label)#cross entropy loss

    X_tain = np.array(X_tain)
    y_train = np.array(y_train)

class Chat_data():
    






