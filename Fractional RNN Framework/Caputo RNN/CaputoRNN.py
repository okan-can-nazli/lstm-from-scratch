import numpy as np
from math import gamma

    
def Caputo(derivs, t, sigma):
    if abs(sigma - 1.0) < 1e-6:
        return derivs[t]
    
    result = 0
    for k in range(t + 1):
        weight = 1.0 if (t - k == 0) else 1.0 / ((t - k) ** sigma)
        result += derivs[k] * weight
    
    return result / gamma(1 - sigma)

class RNNCell:
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu', lr=0.01, sigma = 1):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.sigma = sigma
        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        # Activation
        self._set_activation(activation)
    
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
    
    def forward_sequence(self, x_sequence, h_init=None):

        if h_init is None:
            h_init = np.zeros((1, self.hidden_size))
        
        outputs = []
        caches = []
        h_prev = h_init
        
        for x in x_sequence:
            x = np.atleast_2d(x)  # Ensure (1, input_size)
            
            sum_input = x @ self.W1 + h_prev @ self.W2 
            func_out = self.activation(sum_input + self.b1) 
            y = func_out @ self.W3 + self.b2  
                    
            outputs.append(y)
            caches.append((x, h_prev, func_out))
            
            h_prev = func_out
        
        return outputs, caches
    
    def backward_sequence(self, d_outputs, caches):

        #! Collect standard derivatives
        standard_derivs = []
        for (_, _, func_out) in caches:
            standard_derivs.append(self.activation_deriv(func_out))
        
        #! Apply Caputo
        fractional_derivs = []
        for t in range(len(caches)):
            fractional_derivs.append(
                Caputo(standard_derivs, t, self.sigma)
            )
        
        #! Backprop through time
        dW1 = np.zeros_like(self.W1)
        dW2 = np.zeros_like(self.W2)
        dW3 = np.zeros_like(self.W3)
        db1 = np.zeros_like(self.b1)
        db2 = np.zeros_like(self.b2)
        
        dh_next = np.zeros((1, self.hidden_size))
        
        for t in reversed(range(len(caches))):
            x, h_prev, func_out = caches[t]
            dy = d_outputs[t]
            frac_deriv = fractional_derivs[t]
            
            dW3 += func_out.T @ dy
            db2 += dy
            
            dfunc = dy @ self.W3.T + dh_next
            
            d_sum_input = dfunc * frac_deriv
            
            dW1 += x.T @ d_sum_input
            dW2 += h_prev.T @ d_sum_input
            db1 += d_sum_input
            
            dh_next = d_sum_input @ self.W2.T
        
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.W3 -= self.lr * dW3
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2
