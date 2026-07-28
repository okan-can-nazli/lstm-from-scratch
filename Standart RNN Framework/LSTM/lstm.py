import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return (1 - x**2)

def clip_gradients(grads, max_norm=5.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values() if isinstance(g, np.ndarray)))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for k in grads:
            if isinstance(grads[k], np.ndarray):
                grads[k] *= scale
    return grads

class LSTMCell:
    def __init__(self, input_size, stm_size, output_size=None):
        self.input_size = input_size
        self.stm_size = stm_size
        
        # If output_size is not provided, make it match stm_size
        if output_size is None:
            output_size = stm_size
        
        self.output_size = output_size
        
        # ==========================================
        # 1. CORE LSTM WEIGHTS (The "Brain")
        # Every gate needs weights for BOTH the previous memory AND the new input.
        # So the weight matrix width is: stm_size + input_size
        # Shape: (stm_size rows, stm_size + input_size columns)
        # ==========================================
        
        # Forget Gate: Decides what old memory to delete (0 = forget, 1 = keep)
        self.Wf = np.random.randn(stm_size, stm_size + input_size) * 0.01 
        self.bf = np.ones((stm_size, 1)) # bf.zeros to bf.ones changing
        
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
        # 2. PROJECTION LAYER (The "Translator")
        # Translates the stm_sized-dimensional output into a final prediction format
        # ==========================================

        self.Wy = np.random.randn(self.output_size, stm_size) * 0.1 
        self.by = np.zeros((self.output_size, 1))
    

    def forward(self, x, stm_prev, ltm_prev):
    
        # Step 1: Glue the previous Short-Term Memory and current input together
        # Shape becomes (stm_size + input_size, 1)
        combined_input = np.concatenate([stm_prev, x], axis=0) # matrix merge

        # Step 2: Calculate the 4 Gates
        # Math: Activation( Weights DOT combined_input + bias )
        forget_gate = sigmoid(np.dot(self.Wf, combined_input) + self.bf)
        input_gate = sigmoid(np.dot(self.Wi, combined_input) + self.bi)
        candidate_gate = np.tanh(np.dot(self.Wc, combined_input) + self.bc)
        output_gate = sigmoid(np.dot(self.Wo, combined_input) + self.bo)
        
        # Step 3: Update Long-Term Memory (The "Conveyor Belt")
        # (Old memory * forget gate) + (New memory * input gate)
        ltm_next = forget_gate * ltm_prev + input_gate * candidate_gate
        
        # Step 4: Update Short-Term Memory (What we reveal to the next step)
        stm_next = output_gate * np.tanh(ltm_next)
        
        # Step 5: Take a "Snapshot" for the Backward Pass
        # We save exactly what the network was thinking right now so we can 
        # accurately calculate its mistakes later during BPTT.
        cache = (combined_input, forget_gate, input_gate, candidate_gate, output_gate, ltm_prev, ltm_next, stm_next)

        return stm_next, ltm_next, cache
    
    def forward_sequence(self, x_sequence, stm_init, ltm_init):

        stm_outputs = []
        caches = [] # Our "Photo Album" of snapshots
        
        stm_current = stm_init
        ltm_current = ltm_init
        
        for x in x_sequence:
            x = x.reshape(-1, 1)
            stm_current, ltm_current, cache = self.forward(x, stm_current, ltm_current)
            
            stm_outputs.append(stm_current)
            caches.append(cache)

        return stm_outputs, caches
        
    def backward(self, dstm_next, dltm_next, cache): 
            """
            Backward pass for one timestep
            Calculates gradients using the Chain Rule (dLoss/dTarget = dLoss/dStep * dStep/dTarget)
            """        
            combined_input, forget_gate, input_gate, candidate_gate, output_gate, ltm_prev, ltm_next, _ = cache
            
            # Start with gradient from the future
            dstm = dstm_next
                        
            # ================================================================
            # PART 1: Error of the Memories (STM & LTM)
            # ================================================================
            
            # MATH: dLoss/do = dLoss/dstm * dstm/do
            # Equation: stm = o * tanh(ltm)  =>  Derivative (dstm/do) = tanh(ltm)
            do = dstm * np.tanh(ltm_next)
            
            # MATH: dLoss/dltm = (dLoss/dstm * dstm/dltm) + dltm_next
            # Equation: stm = o * tanh(ltm)  =>  Derivative (dstm/dltm) = o * (1 - tanh^2(ltm))
            dltm = dstm * output_gate * tanh_derivative(np.tanh(ltm_next))
            dltm += dltm_next  # TOTAL dLoss/dltm
            
            # ================================================================
            # PART 2: Error of the Internal Gates (Splitting LTM error)
            # Equation: ltm = (f * ltm_prev) + (i * c)
            # ================================================================

            # MATH: dLoss/df = dLoss/dltm * dltm/df
            # Derivative (dltm/df) = ltm_prev
            df = dltm * ltm_prev
            
            # MATH: dLoss/dltm_prev = dLoss/dltm * dltm/dltm_prev
            # Derivative (dltm/dltm_prev) = f
            dltm_prev = dltm * forget_gate
            
            # MATH: dLoss/di = dLoss/dltm * dltm/di
            # Derivative (dltm/di) = c (candidate_gate)
            di = dltm * candidate_gate       
            
            # MATH: dLoss/dc = dLoss/dltm * dltm/dc
            # Derivative (dltm/dc) = i (input_gate)
            dc = dltm * input_gate
            
            # ================================================================
            # PART 3: Pass Error Through Activation Functions
            # ================================================================
            
            # MATH: dLoss/do_input = dLoss/do * Derivative of Sigmoid(o_input)
            # Sigmoid Derivative = sig * (1 - sig)
            do_input = do * sigmoid_derivative(output_gate)
            
            # MATH: dLoss/di_input = dLoss/di * Derivative of Sigmoid(i_input)
            di_input = di * sigmoid_derivative(input_gate)
            
            # MATH: dLoss/df_input = dLoss/df * Derivative of Sigmoid(f_input)
            df_input = df * sigmoid_derivative(forget_gate)
            
            # MATH: dLoss/dc_input = dLoss/dc * Derivative of Tanh(c_input)
            # Tanh Derivative = 1 - x^2 (candidate_gate is already tanh'd, so this is correct now)
            dc_input = dc * tanh_derivative(candidate_gate)
            
            # ================================================================
            # PART 4: Calculate Weight and Bias Gradients
            # Equation: gate_input = (W * combined_input) + b
            # ================================================================
            
            # MATH: dLoss/dWo = dLoss/do_input * do_input/dWo
            # Derivative (do_input/dWo) = combined_input.T
            dWo = np.dot(do_input, combined_input.T)
            dbo = do_input
            
            dWi = np.dot(di_input, combined_input.T)
            dbi = di_input
            
            dWf = np.dot(df_input, combined_input.T)
            dbf = df_input
            
            dWc = np.dot(dc_input, combined_input.T)
            dbc = dc_input
            
            # ================================================================
            # PART 5: Gradient for Combined Input (To pass backward to x and stm_prev)
            # ================================================================


            # MATH: dLoss/dcombined = dLoss/dcombined_via_f + dLoss/dcombined_via_i + dLoss/dcombined_via_c + dLoss/dcombined_via_o
            
            # Each term:
            # dLoss/dcombined_via_o = dLoss/do_input * do_input/dcombined = Wo.T * do_input
            # dLoss/dcombined_via_i = dLoss/di_input * di_input/dcombined = Wi.T * di_input
            # dLoss/dcombined_via_f = dLoss/df_input * df_input/dcombined = Wf.T * df_input
            # dLoss/dcombined_via_c = dLoss/dc_input * dc_input/dcombined = Wc.T * dc_input
            
            # Total = sum all paths
            
            # MATH: dLoss/dcombined = Sum of (Weight.T * dgate_input) for all 4 gates

            dcombined_input = (np.dot(self.Wo.T, do_input) +
                        np.dot(self.Wi.T, di_input) +
                        np.dot(self.Wf.T, df_input) +
                        np.dot(self.Wc.T, dc_input))
            

            # Split combined input gradient back into STM and X
            dstm_prev = dcombined_input[:self.stm_size]
            dx = dcombined_input[self.stm_size:] # we use raw data so its unnecessary but better to declare cause why not
            
            return {
                'dWo': dWo, 'dbo': dbo,
                'dWi': dWi, 'dbi': dbi,
                'dWf': dWf, 'dbf': dbf,
                'dWc': dWc, 'dbc': dbc,
                'dstm_prev': dstm_prev,
                'dltm_prev': dltm_prev,
                'dx': dx
            }

    def backward_sequence(self, dy_preds, stm_outputs, caches):
        """
        BPTT for many-to-many tasks. 
        dy_preds: A LIST of errors, one for each timestep.
        """
        # 1. Initialize empty buckets for all gradients
        accumulated_grads = {
            'dWo': np.zeros_like(self.Wo), 'dbo': np.zeros_like(self.bo),
            'dWi': np.zeros_like(self.Wi), 'dbi': np.zeros_like(self.bi),
            'dWf': np.zeros_like(self.Wf), 'dbf': np.zeros_like(self.bf),
            'dWc': np.zeros_like(self.Wc), 'dbc': np.zeros_like(self.bc),
            'dWy': np.zeros_like(self.Wy), 'dby': np.zeros_like(self.by)
        }
        
        # Start with zero "future" error
        dstm_next = np.zeros((self.stm_size, 1))
        dltm_next = np.zeros((self.stm_size, 1))
        
        # 2. Walk backwards through time
        for t in reversed(range(len(caches))):
            dy_t = dy_preds[t]      # How wrong the prediction was at step 't'
            stm_t = stm_outputs[t]  # The STM that was used to make that prediction
            
            # dWy = Error * Input (transposed for matrix math)
            accumulated_grads['dWy'] += np.dot(dy_t, stm_t.T)

            # dby = Error (Bias gradient is just the raw error)
            accumulated_grads['dby'] += dy_t
            
            
            # Add the Mouth error to the Brain error (Error from Future + Error from now)
            dstm_this_step = dstm_next + np.dot(self.Wy.T, dy_t)
            
            # Run the standard backward pass for this step
            grads = self.backward(dstm_this_step, dltm_next, caches[t])
            
            # Accumulate Brain gradients
            for key in ['dWo', 'dbo', 'dWi', 'dbi', 'dWf', 'dbf', 'dWc', 'dbc']:
                accumulated_grads[key] += grads[key]
            
            # Hand the baton to the previous step
            dstm_next = grads['dstm_prev']
            dltm_next = grads['dltm_prev']
            
        return accumulated_grads

    def update_weights(self, grads, learning_rate=0.01, max_norm=5.0):
        """
        We use .get('key', 0) so it doesn't crash if a gradient is missing. 
        """
        grads = clip_gradients(grads, max_norm=max_norm) # normalization

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