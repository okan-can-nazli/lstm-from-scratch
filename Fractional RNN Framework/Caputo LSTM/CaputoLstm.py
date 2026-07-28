import numpy as np
from math import gamma 

def sigmoid(x):
    """Squashes numbers between 0 and 1"""
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return (1 - x**2)


"""
Continuous form:
D^sigma f(t) = (1/Γ(1-sigma)) ∫₀ᵗ f'(τ)/(t-τ)^sigma dτ
Discrete form (what we use in code):
D^sigma f(t) = (1/Γ(1-sigma)) * sigma[k=0 to t] f'(k)/(t-k)^sigma
"""

def Caputo(standart_derivatives,sigma,current_t):
    
    caputo_grad = 0
    
    if abs(sigma - 1.0) < 0.00001 : return standart_derivatives[current_t] # return standart derivative for such a low affect rate
    
    for k in range(current_t + 1):
        
        
        if current_t - k == 0:
            weight = 1
        else:
            weight = 1 / ((current_t - k ) ** sigma)
    
    
        caputo_grad += standart_derivatives[k] * weight
    
    return 1/gamma(1-sigma) * caputo_grad
    


class LSTMCell:
    def __init__(self, input_size, stm_size, output_size=None):
        self.input_size = input_size
        self.stm_size = stm_size
        
        # If output_size is not provided, make it match stm_size
        if output_size is None:
            output_size = stm_size
        
        self.output_size = output_size
        
        # ==========================================
        # 1. CORE LSTM WEIGHTS
        # Shape: (stm_size rows, stm_size + input_size columns)
        # ==========================================
        
        # Forget Gate: Decides what old memory to delete (0 = forget, 1 = keep)
        self.Wf = np.random.randn(stm_size, stm_size + input_size) * 0.01 
        self.bf = np.zeros((stm_size, 1))
        
        # Input Gate: Decides which parts of the new memory to let in
        self.Wi = np.random.randn(stm_size, stm_size + input_size) * 0.01
        self.bi = np.zeros((stm_size, 1))

        # candidate_gate (New Potential Memory): The actual new information we might learn
        self.Wc = np.random.randn(stm_size, stm_size + input_size) * 0.01
        self.bc = np.zeros((stm_size, 1))
        
        # Output Gate: Decides what part of the internal memory to reveal to the world
        self.Wo = np.random.randn(stm_size, stm_size + input_size) * 0.01
        self.bo = np.zeros((stm_size, 1))
        
        # ==========================================
        # 2. PROJECTION LAYER
        # Translates the stm_sized output into a final prediction sized format
        # ==========================================

        self.Wy = np.random.randn(self.output_size, stm_size) * 0.1 
        self.by = np.zeros((self.output_size, 1))
    
    
    def forward(self, x, stm_prev, ltm_prev):
        """
        Takes one single step forward in time.
        """
        # Step 1: Glue the previous Short-Term Memory and current input together
        # Shape becomes (stm_size + input_size, 1)
        combined_input = np.concatenate([stm_prev, x], axis=0) # matrix merge

        # Step 2: Calculate the 4 Gates
        # Math: Activation( Weights DOT combined_input + bias )
        forget_gate = sigmoid(np.dot(self.Wf, combined_input) + self.bf)
        input_gate = sigmoid(np.dot(self.Wi, combined_input) + self.bi)
        candidate_gate = np.tanh(np.dot(self.Wc, combined_input) + self.bc)
        output_gate = sigmoid(np.dot(self.Wo, combined_input) + self.bo)
        
        # Step 3: Update Long-Term Memory
        # (Old memory * forget gate) + (New memory * input gate)
        ltm_next = forget_gate * ltm_prev + input_gate * candidate_gate
        
        # Step 4: Update Short-Term Memory (What we reveal to the next step)
        stm_next = output_gate * np.tanh(ltm_next)
        
        # Step 5: Take a "Snapshot" for the Backward Pass
        # We save exactly what the network was thinking right now so we can 
        # accurately calculate its mistakes later during BPTT.
        cache = (combined_input, forget_gate, input_gate, candidate_gate, output_gate, ltm_prev, ltm_next, stm_next) # x, stm_prev, predictiion may be seem not redundant but better for understanding #! I removed them cause these vars literally does not fit forward method logic

        return stm_next, ltm_next, cache
    
    def forward_sequence(self, x_sequence, stm_init, ltm_init):

        stm_outputs = []
        caches = [] # Our "Photo Album" of snapshots
        
        stm_current = stm_init
        ltm_current = ltm_init
        
        for x in x_sequence:
            x = x.reshape(-1, 1)
            stm_current, ltm_current, cache = self.forward(x, stm_current, ltm_current)
            
            caches.append(cache)
            stm_outputs.append(stm_current)

        return stm_outputs, caches
        

    def backward_sequence(self, dy_preds, stm_outputs, caches, sigma=1):
        accumulated_grads = {
            'dWo': np.zeros_like(self.Wo), 'dbo': np.zeros_like(self.bo),
            'dWi': np.zeros_like(self.Wi), 'dbi': np.zeros_like(self.bi),
            'dWf': np.zeros_like(self.Wf), 'dbf': np.zeros_like(self.bf),
            'dWc': np.zeros_like(self.Wc), 'dbc': np.zeros_like(self.bc),
            'dWy': np.zeros_like(self.Wy), 'dby': np.zeros_like(self.by)
        }
        
        #! Collect standard derivatives
        standard_derivs = {
            'output_gate': [],
            'input_gate': [],
            'forget_gate': [],
            'candidate_gate': []
        }
        
        for cache in caches:
            
            _, forget_gate, input_gate, candidate_gate, output_gate, _, _, _ = cache
            
            standard_derivs['output_gate'].append(sigmoid_derivative(output_gate))
            standard_derivs['input_gate'].append(sigmoid_derivative(input_gate))
            standard_derivs['forget_gate'].append(sigmoid_derivative(forget_gate))
            standard_derivs['candidate_gate'].append(tanh_derivative(candidate_gate))
            
        
        #! Apply Caputo 
        fractional_derivs = {
            'output_gate': [],
            'input_gate': [],
            'forget_gate': [],
            'candidate_gate': []
        }
        
        for t in range(len(caches)):
            fractional_derivs['output_gate'].append(
                Caputo(standard_derivs['output_gate'], sigma, t)
            )
            fractional_derivs['input_gate'].append(
                Caputo(standard_derivs['input_gate'], sigma, t)
            )
            fractional_derivs['forget_gate'].append(
                Caputo(standard_derivs['forget_gate'], sigma, t)
            )
            fractional_derivs['candidate_gate'].append(
                Caputo(standard_derivs['candidate_gate'], sigma, t)
            )
        
        #! Backward pass using fractional derivatives
        dstm_next = np.zeros((self.stm_size, 1))
        dltm_next = np.zeros((self.stm_size, 1))
        
        for t in reversed(range(len(caches))):
            cache = caches[t]
            combined_input, forget_gate, input_gate, candidate, output_gate, ltm_prev, ltm_next, _ = cache            
            dy_t = dy_preds[t]
            stm_t = stm_outputs[t]
            
            # Projection layer gradients
            accumulated_grads['dWy'] += np.dot(dy_t, stm_t.T)
            accumulated_grads['dby'] += dy_t
            
            # STM error
            dstm = dstm_next + np.dot(self.Wy.T, dy_t)
            
            # Memory errors (same as standard)
            do = dstm * np.tanh(ltm_next)
            dltm = dstm * output_gate * tanh_derivative(np.tanh(ltm_next)) + dltm_next
            
            df = dltm * ltm_prev
            dltm_prev = dltm * forget_gate
            di = dltm * candidate
            dc = dltm * input_gate
            
            #! === USE FRACTIONAL DERIVATIVES HERE ===
            do_input = do * fractional_derivs['output_gate'][t]
            di_input = di * fractional_derivs['input_gate'][t]
            df_input = df * fractional_derivs['forget_gate'][t]
            dc_input = dc * fractional_derivs['candidate_gate'][t]
            
            # Weight gradients
            accumulated_grads['dWo'] += np.dot(do_input, combined_input.T)
            accumulated_grads['dbo'] += do_input
            accumulated_grads['dWi'] += np.dot(di_input, combined_input.T)
            accumulated_grads['dbi'] += di_input
            accumulated_grads['dWf'] += np.dot(df_input, combined_input.T)
            accumulated_grads['dbf'] += df_input
            accumulated_grads['dWc'] += np.dot(dc_input, combined_input.T)
            accumulated_grads['dbc'] += dc_input
            
            # Error for previous timestep
            dcombined = (np.dot(self.Wo.T, do_input) +
                        np.dot(self.Wi.T, di_input) +
                        np.dot(self.Wf.T, df_input) +
                        np.dot(self.Wc.T, dc_input))
            
            dstm_next = dcombined[:self.stm_size]
            dltm_next = dltm_prev
        
        return accumulated_grads

    def update_weights(self, grads, learning_rate=0.01):
        """
        We use .get('key', 0) so it doesn't crash if a gradient is missing. 
        """
        self.Wo -= learning_rate * grads.get('dWo', 0)
        self.bo -= learning_rate * grads.get('dbo', 0)
        
        self.Wi -= learning_rate * grads.get('dWi', 0)
        self.bi -= learning_rate * grads.get('dbi', 0)
        
        self.Wf -= learning_rate * grads.get('dWf', 0)
        self.bf -= learning_rate * grads.get('dbf', 0)
        
        self.Wc -= learning_rate * grads.get('dWc', 0)
        self.bc -= learning_rate * grads.get('dbc', 0)

        # Update the projection layer weights
        self.Wy -= learning_rate * grads.get('dWy', 0)
        self.by -= learning_rate * grads.get('dby', 0)
        
    def predict(self, stm):
        """Translates a hidden state into a final prediction"""
        return np.dot(self.Wy, stm) + self.by