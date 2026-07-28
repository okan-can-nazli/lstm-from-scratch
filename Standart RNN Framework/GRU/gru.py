import numpy as np

# functions
from numpy import tanh
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
def tanh_derivative(x):
    return (1 - x ** 2)

class GRUCell:
    def __init__(self, input_size, hidden_size, output_size = None):
        if output_size is None:
            output_size = hidden_size
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        #Lets try a seperated weight alternative instead of we have done in lstm code
        
        #Basiccaly we convert each data to a (1,hidden) format at the end of the process
        #Weighs initialize
        
        """
        u : update (z)
        n : new (candidates)
        i : input_data
        h : hidden_data
        """
        
        #Weights initialize
        w_ir = np.random.randn(input_size,hidden_size) * 0.01 
        w_hr = np.random.randn(hidden_size, hidden_size) * 0.01 
        w_iu = np.random.randn(input_size, hidden_size) * 0.01 
        w_hu = np.random.randn(hidden_size, hidden_size) * 0.01 
        w_in = np.random.randn(input_size, hidden_size) * 0.01 
        w_hn = np.random.randn(hidden_size, hidden_size) * 0.01 

        
        #Biases initialize
        b_ir = np.zeros((1, hidden_size)) 
        b_hr = np.zeros((1, hidden_size)) 
        b_iu = np.zeros((1, hidden_size)) 
        b_hu = np.zeros((1, hidden_size)) 
        b_in = np.zeros((1, hidden_size)) 
        b_hn = np.zeros((1, hidden_size))
        
        

        # projection layer does not included in loss calculation 
        w_y = np.random.randn(hidden_size, output_size) * 0.01
        b_y = np.zeros((1, self.output_size))
        
        
        
        weights = {
            "w_ir" : w_ir,
            "w_hr" : w_hr,
            "w_iu" : w_iu,
            "w_hu" : w_hu,
            "w_in" : w_in,
            "w_hn" : w_hn,
            
            "w_y" : w_y
        }
        
        self.weights = weights

        
        biases = {
            "b_ir" : b_ir,
            "b_hr" : b_hr,
            "b_iu" : b_iu,
            "b_hu" : b_hu,
            "b_in" : b_in,
            "b_hn" : b_hn,
            
            "b_y" : b_y
        }
        
        self.biases = biases
        
        
    def forward(self, x, h_prev):
        
                                                                        # GATES
                                                                        
        reset_gate = sigmoid(x @ self.weights["w_ir"]  + self.biases["b_ir"] + h_prev @ self.weights["w_hr"] + self.biases["b_hr"])
        update_gate = sigmoid(x @ self.weights["w_iu"] + self.biases["b_iu"] + h_prev @ self.weights["w_hu"] + self.biases["b_hu"])
        new_gate = tanh(x @ self.weights["w_in"] + self.biases["b_in"] + (reset_gate * h_prev) @ self.weights["w_hn"] + self.biases["b_hn"])
        
        #! WE GONNA USE THİS EQUATİON ON GATE LOSS DECLERATİONS
        h_next = (1 - update_gate) * new_gate + update_gate * h_prev
        

        y = h_next @ self.weights["w_y"] + self.biases["b_y"]



        self.last_cache = (x, h_prev, reset_gate, update_gate, new_gate, h_next, y)
        
        return h_next, y
    
    
    # x_sequence is a grouped part from all data 
    def forward_sequence(self, x_sequence, h_init = None):
        
        if h_init is None:
            h_init = np.zeros((1, self.hidden_size))
            
        h_prev = h_init
        
        all_y = []
        caches = []
        
        for x in x_sequence:
            x = x.reshape(1, -1) # force the x_list to become a single-row format
            h_next, y = self.forward(x, h_prev)
            all_y.append(y)
            caches.append(self.last_cache)
            h_prev = h_next
        self.caches = caches
        
        return np.vstack(all_y) # vertically stacked format return
    
    
    def backward(self, d_outputs, lr = 0.01):
    
        accumulated_grads = {
            "w_ir": np.zeros_like(self.weights["w_ir"]),
            "w_hr": np.zeros_like(self.weights["w_hr"]),
            "w_iu": np.zeros_like(self.weights["w_iu"]),
            "w_hu": np.zeros_like(self.weights["w_hu"]),
            "w_in": np.zeros_like(self.weights["w_in"]),
            "w_hn": np.zeros_like(self.weights["w_hn"]),
            "b_ir": np.zeros_like(self.biases["b_ir"]),
            "b_hr": np.zeros_like(self.biases["b_hr"]),
            "b_iu": np.zeros_like(self.biases["b_iu"]),
            "b_hu": np.zeros_like(self.biases["b_hu"]),
            "b_in": np.zeros_like(self.biases["b_in"]),
            "b_hn": np.zeros_like(self.biases["b_hn"]),
            
            "w_y": np.zeros_like(self.weights["w_y"]),
            "b_y": np.zeros_like(self.biases["b_y"])
        }
        
            #!!! THE CHAIN RULE !!! 
        
        
        d_h_next = np.zeros((1, self.hidden_size))  # d_h_next INIT

        for t in reversed(range(len(self.caches))):
             
            x, h_prev, reset_gate, update_gate, new_gate, h_next, y = self.caches[t]
            
            d_y = d_outputs[t].reshape(1,-1)

            dh = (d_y @ self.weights["w_y"].T) + d_h_next #!ALL H  LOSS


                        
            
            
            
            #dLOSS/dnew_gate = dLOSS/dh * dh/dnew_gate
            d_new_gate_input = dh * (1-update_gate) * tanh_derivative(new_gate)
            
            #dLOSS/update_gate = dLOSS/dh * dh/dupdate_gate
            d_update_gate_input = dh * (h_prev - new_gate) * sigmoid_derivative(update_gate)
            
            #dLOSS/d_reset_gate = dLOSS/dh * dh/dnew_gate * dnew_gate/dreset_gate
            d_reset_gate_input = (d_new_gate_input @ self.weights["w_hn"].T) * h_prev * sigmoid_derivative(reset_gate)            
            
            ###########################################################################################################
            
                                    # NEW GATE LOSSES
            
            #dLOSS/dw_in = dLOSS/d_new_gate_input * d_new_gate_input/dw_in
            dw_in = x.T @ d_new_gate_input
            
            #dLOSS/dw_hn = dLOSS/d_new_gate_input * d_new_gate_input/dw_hn
            dw_hn = (reset_gate * h_prev).T @ d_new_gate_input            
            #dLOSS/db_in = dLOSS/d_new_gate_input * d_new_gate_input/db_in
            db_in = d_new_gate_input

            #dLOSS/db_hn = dLOSS/d_new_gate_input * d_new_gate_input/db_hn
            db_hn = d_new_gate_input

            
                                    # UPDATE GATE LOSSES
            
            #dLOSS/dw_iu = dLOSS/d_update_gate_input * d_update_gate_input/dw_iu
            dw_iu = x.T @ d_update_gate_input
            
            #dLOSS/dw_hu = dLOSS/d_update_gate_input * d_update_gate_input/dw_hu
            dw_hu = h_prev.T @ d_update_gate_input
            
            #dLOSS/db_iu = dLOSS/d_update_gate_input * d_update_gate_input/db_iu
            db_iu = d_update_gate_input
            
            #dLOSS/db_hu = dLOSS/d_update_gate_input * d_update_gate_input/db_hu
            db_hu = d_update_gate_input


                                    # RESET GATE LOSSES

            #dLOSS/dw_ir = dLOSS/d_reset_gate_input * d_reset_gate_input/dw_iu
            dw_ir = x.T @ d_reset_gate_input
            
            #dLOSS/dw_hr = dLOSS/d_reset_gate_input * d_reset_gate_input/dw_hu
            dw_hr = h_prev.T @ d_reset_gate_input
            
            #dLOSS/db_ir = dLOSS/d_reset_gate_input * d_reset_gate_input/db_ir
            db_ir = d_reset_gate_input
            
            #dLOSS/db_hr = dLOSS/d_reset_gate_input * d_reset_gate_input/db_hu
            db_hr = d_reset_gate_input

    
            
            # Calculate projection layer's loss
            dw_y = h_next.T @ d_y
            db_y = d_y 
            
            # --- ACCUMULATE PROJECTION LAYER ---
            accumulated_grads["w_y"] += dw_y
            accumulated_grads["b_y"] += db_y
            
            
            # --- ACCUMULATE NEW GATE ---
            accumulated_grads["w_in"] += dw_in
            accumulated_grads["w_hn"] += dw_hn
            accumulated_grads["b_in"] += db_in
            accumulated_grads["b_hn"] += db_hn

            # --- ACCUMULATE UPDATE GATE ---
            accumulated_grads["w_iu"] += dw_iu
            accumulated_grads["w_hu"] += dw_hu
            accumulated_grads["b_iu"] += db_iu
            accumulated_grads["b_hu"] += db_hu

            # --- ACCUMULATE RESET GATE ---
            accumulated_grads["w_ir"] += dw_ir
            accumulated_grads["w_hr"] += dw_hr
            accumulated_grads["b_ir"] += db_ir
            accumulated_grads["b_hr"] += db_hr
        
        




            d_h_prev = (dh * update_gate) + \
                (d_new_gate_input @ self.weights["w_hn"].T) * reset_gate + \
                (d_update_gate_input @ self.weights["w_hu"].T) + \
                (d_reset_gate_input @ self.weights["w_hr"].T)
        
            d_h_next = d_h_prev
        
        
        
        
        
                 #UPDATE WEİGHTS-BİASES
                 
        for key in self.weights:
            self.weights[key] -= lr * accumulated_grads[key]
            
        for key in self.biases:
            self.biases[key] -= lr * accumulated_grads[key]
        
        
        
        
               