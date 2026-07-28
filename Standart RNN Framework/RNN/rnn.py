import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size, output_size=None, activation="relu", lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        if output_size is None:
            output_size = hidden_size
            
        self.output_size = output_size
        self.lr = lr
        self._set_activation(activation)
        
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01 
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.randn(hidden_size, hidden_size) * 0.01

        self.w3 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        

    def _set_activation(self, name):
        """Set activation function and its derivative."""
        if name == 'relu':
            self.activation = lambda x: np.maximum(0, x) 
            self.activation_deriv = lambda h: (h > 0).astype(float)
        elif name == 'tanh':
            self.activation = np.tanh
            self.activation_deriv = lambda h: 1 - h**2
        elif name == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_deriv = lambda h: h * (1 - h)
        else:
            raise ValueError(f"Unknown activation: {name}")
        
        
    def forward(self, x, h_prev):
        sum_input = x @ self.w1 + h_prev @ self.w2 
        func_out = self.activation(sum_input + self.b1) 
        y = func_out @ self.w3 + self.b2  
        
        cache = (x, h_prev, sum_input, func_out, y)        
        self.cache = cache

        h_next = func_out 
        
        return h_next, y
    
    
    def forward_sequence(self, x_sequence, h_init=None):
        if h_init is None:
            h_init = np.zeros((1, self.hidden_size))
            
        h_prev = h_init
        all_y = []
        caches = []
        for x in x_sequence:

            x = x.reshape(1, -1) 
            h_prev, y = self.forward(x, h_prev)
            all_y.append(y)
            caches.append(self.cache)
        
        self.caches = caches
            
        return all_y
    
    def backward(self, d_out, dh_next, cache):
        x, h_prev, sum_input, func_out, y = cache
        
        # Calculate gradient from the output layer for this specific timestep
        dh_from_y = d_out @ self.w3.T
        
        # Combine with the gradient flowing from the future
        dh_total = dh_from_y + dh_next
        
        # Backpropagate through the activation
        dfunc = dh_total * self.activation_deriv(func_out)   
        
        dx = dfunc @ self.w1.T        
        dh_prev = dfunc @ self.w2.T        
        dw1 = x.T @ dfunc       
        dw2 = h_prev.T @ dfunc             
        dw3 = func_out.T @ d_out    
        db1 = dfunc
        db2 = d_out                  

        self.grads["w1"].append(dw1)
        self.grads["w2"].append(dw2)
        self.grads["w3"].append(dw3)
        self.grads["b1"].append(db1)
        self.grads["b2"].append(db2)

        return dh_prev
    
    
    def backward_sequence(self, d_outputs):
        grads = {
            "w1" : [],
            "w2" : [],
            "w3" : [],
            "b1" : [],
            "b2" : [],
        }
        self.grads = grads

        dh_next = np.zeros((1, self.hidden_size))
        
       # Pass d_out and dh_next as distinct arguments
        for d_out, cache in zip(reversed(d_outputs), reversed(self.caches)):
            dh_next = self.backward(d_out, dh_next, cache)
        
        self.update_weights()
            
    def update_weights(self):
        self.w1 -= self.lr * sum(self.grads["w1"])
        self.w2 -= self.lr * sum(self.grads["w2"])
        self.w3 -= self.lr * sum(self.grads["w3"])
        self.b1 -= self.lr * sum(self.grads["b1"])
        self.b2 -= self.lr * sum(self.grads["b2"])