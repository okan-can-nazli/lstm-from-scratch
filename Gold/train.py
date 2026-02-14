import torch
import torch.nn as nn
import torch.optim as optim


def train_model(lstm,prices_list):
    price_t = torch.tensor(prices_list).float().unfold(0, 30, 1) #breaks prices into sliding windows (dimension,window size,slide range) [[1,2,3,....30],[2,3,4...31],...]
        
        
        
    criterion = nn.MSELoss() #Mean Squared Error 

    optimizer = optim.Adam(lstm.parameters(), lr = 0.001) #yes.It is a high lr ,isnt it?
    #weight update algorithm (momentum etc.)
        
        
    print("Training started")
    lstm.train() # NOT neccessary for THIS project
    #start train method (Dropout, BatchNorm)
    #Dropouts: Shut down random neurons to prevent overfitting (chaos monkey)
    #Batch Normalization: Keep values in a good range so gradients dont explode/vanish



    for window in price_t:
        
        optimizer.zero_grad()#clear old gradients
        
        
        #-1 mean "figure out dimension yourself"
        x_t = window[:-1].view(1, -1 ,1) #reshapes (29,) to (1, 29, 1)        ValueError: LSTM: Expected input to be 2D or 3D, got 1D instead
        y_t = window[-1].view(1,1)
        
        #forward
        prediction = lstm(x_t)
        
        #loss
        loss = criterion(prediction, y_t)
        
        #backward
        loss.backward()
        optimizer.step()
        
    lstm.eval()
    #end of the train method

    print("Train successful")

    #LSTM exits to learn from sequences,keep in mind

