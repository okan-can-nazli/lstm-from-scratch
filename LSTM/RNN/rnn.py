import numpy as np

class RNNCell:
    def __init__(self,input_size, hidden_size):
        
        # x.shape = (1,input_size)
        # hidden.shape = (1,hidden_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.w1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1,hidden_size))
        
        self.w2 = np.random.rand(hidden_size,hidden_size)

        self.w3 = np.random.rand(hidden_size,hidden_size)
        self.b2 = np.zeros(1,hidden_size)
    
        #I could also initialize hidden_state however it is better to get it from user too
        
        
    # ACTİVATİON FUNCTİONS
    
    def ReLU(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def forward(self,x, h_prev):
        
        # I prefered "seperate weight application method" (not that much efficient but cleaner)
        
        sum_input = np.dot(x, self.w1) + np.dot(h_prev, self.w2)
        
        h_next = self.ReLU(sum_input) # It is also output of activation gate
        step_outputs = np.dot(h_next, self.w3) + self.b2
        
        self.h_next = h_next
        self.step_outputs = step_outputs
        
        return h_next, step_outputs
    
    
    def forward_sequence(self,x_sequence, hidden_init):
        h_prev = hidden_init
        all_outputs = []

        for x in x_sequence:
            x = x.reshape(-1,1)
            h_prev, step_outputs = self.forward(x, h_prev)
            
            all_outputs.append(step_outputs)
            
        return all_outputs
    
    def backward(self, d_all_outputs):
        
        dw1 =
        dw2 =
        dw3 =
        db1 =
        db2 = 
        