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
        self.b2 = np.zeros((1,hidden_size))
    
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
        
        sum_input = np.dot(x, self.w1) + np.dot(h_prev, self.w2) # notice that we used these equations on backward
        
        func_out = self.ReLU(sum_input + self.b1) # It is also output of activation gate
        step_outputs = np.dot(func_out, self.w3) + self.b2   # notice that we used these equations on backward
        
        self.func_out = func_out
        self.step_outputs = step_outputs
        
        self.x = x
        self.h_prev = h_prev
        self.sum_input = sum_input
        
        return func_out, step_outputs
    
    
    def forward_sequence(self,x_sequence, hidden_init):
        h_prev = hidden_init
        all_outputs = []

        for x in x_sequence:
            x = x.reshape(-1,1)
            h_prev, step_outputs = self.forward(x, h_prev)
            
            all_outputs.append(step_outputs)
            
        return all_outputs
    
    def backward(self, d_all_outputs):
        
        d_out_next = d_all_outputs
        
        
        # output = (ReLU(x * w1 + h_prev * w2 + b1)* w3) + b2
        # apply ! THE CHAIN RULE !
        
        dfunc = np.dot(d_out_next, self.w3.T)     #TRANSPOSE, HOW DİDNT U EVEN NOTİCE THAT?!  |    d_out_next * self.w3 . T  !!!!
        
        dx = np.dot(dfunc, self.w1.T)         #  dfunc * self.w2 . T
        dh = np.dot(dfunc, self.w2.T)               #  dfunc * self.w2 . T
        dw1 = np.dot(self.x.T,dfunc)         # dfunc * x
        dw2 = np.dot(self.h_prev.T,dfunc)              # dfunc * h
        dw3 = np.dot(self.func_out.T,d_out_next)
        db1 = (self.sum_input > 0) * dfunc
        db2 = d_out_next
        
    
    
    
    

"""
Simple rule:

In forward pass: output = np.dot(A, B)

In backward pass:

Gradient for A = np.dot(grad, B.T)
Gradient for B = np.dot(A.T, grad)
Left side of dot → use right.T Right side of dot → use left.T



step_outputs = np.dot(func_out, self.w3)

func_out is left → gradient = np.dot(grad, self.w3.T) ✅ (your dfunc)
self.w3 is right → gradient = np.dot(func_out.T, grad) (your dw3)

sum_input = np.dot(x, self.w1)

x is left → gradient = np.dot(dfunc, self.w1.T) ✅ (your dx)
self.w1 is right → gradient = np.dot(x.T, dfunc) (your dw1)"""