import numpy as np

class RNNCell:
    def __init__(self,input_size, hidden_size):
        
        # x.shape = (1,input_size)
        # hidden.shape = (1,hidden_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01 
        self.b1 = np.zeros((1,hidden_size))
        
        self.w2 = np.random.randn(hidden_size,hidden_size) * 0.01

        self.w3 = np.random.randn(hidden_size,hidden_size) * 0.01
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
        
        # pass vars to backward method 
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
        
        h_next = self.h_prev # h from next cell
        
        d_out_next = d_all_outputs
        
        
        # output = (ReLU(x * w1 + h_prev * w2 + b1) * w3) + b2
        # apply ! THE CHAIN RULE !
        
        dfunc = np.dot(d_out_next, self.w3.T) * (self.sum_input>0)    #TRANSPOSE, HOW DİDNT U EVEN NOTİCE THAT?!  
        
        dx = np.dot(dfunc, self.w1.T)        
        dh_next = np.dot(dfunc, self.w2.T)        
        dw1 = np.dot(self.x.T,dfunc)         
        dw2 = np.dot(h_next.T,dfunc)             
        dw3 = np.dot(self.func_out.T,d_out_next)
        db1 = dfunc
        db2 = d_out_next
        
    
        # TRANSPOSE RULE: not about math, about making shapes work
        
        # Forward:  np.dot(A, B)
        # Backward: 
        #           dA = np.dot(grad, B.T)  → B.T flips shape so result matches A
        #           dB = np.dot(A.T, grad)  → A.T flips shape so result matches B
        

        lr = 0.01
        self.w1 -= lr * dw1
        self.w2 -= lr * dw2
        self.w3 -= lr * dw3
        self.b1 -= lr * db1
        self.b2 -= lr * db2
        
        return dh_next
    
    
    def backward_sequence(self,d_outputs):
        dh_next = np.zeros((1, self.hidden_size))
        
        for d_out in reversed(d_outputs):
            dh_next = self.backward(d_out + dh_next)