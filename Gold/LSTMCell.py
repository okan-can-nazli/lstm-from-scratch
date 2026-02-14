
import torch
import torch.nn as nn
import torch.optim as optim

class GPP(nn.Module): #basic syntax of PyTorch
    def __init__(self):
        
        super().__init__() #calls nn.Module's (parent class) __init__  / Basic Syntax
       
        self.lstm = nn.LSTM(input_size=1,hidden_size=64,batch_first=True) #initialization of weights,gradients,optimization etc.
        self.output_layer = nn.Linear(64,1) #we expect 1 output from 64 hiddens
        
    def forward(self,x): #29 data enters once 
        
        out , _ = self.lstm(x) # x.shape()=  1 sequence at a time,29 timesteps,1 feature
        #out = each stm_output,   "_" = last stm and ltm output value (named as "_" because we dont care it D:)
        
        return self.output_layer(out[:, -1, :])  #(1,64) @ (64,1)*wo + bo
    