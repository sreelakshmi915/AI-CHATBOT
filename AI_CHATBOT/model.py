# we using feed forward neural network
import torch
import torch.nn as nn
class Neuralnetwork(nn.Module): # creating neural network with 3 layers input hidden and output
    def __init__(self, input_size, hidden_size, num_classes):
        super(Neuralnetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)#input layer here iput to 1st layer is input_size and output to hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)#hidden layer here iput tohidden layer is hidden_size and output to hidden
        self.l3 = nn.Linear(hidden_size, num_classes)#output layer here iput to outputlayer ishidden_size and output is num_classes
        #input size and num _classes fixed but hiddden_size can be change
        self.relu = nn.ReLU()
    def forward(self,x):#fn for forward pass
        out = self.l1(x) #input to l1
        out = self.relu(out) #activation fn to that node
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        #last we dont want activation fn
        return out


