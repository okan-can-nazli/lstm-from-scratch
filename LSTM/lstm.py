import numpy as np

class LSTMCell:
    def __init__(self, input_size, stm_size):
        
        self.input_size = input_size # line format
        """
        The Input (x):
        
            Size: 3 (Pizza, Rain, Sleep)
            Shape: A vertical line (Vector)
            Data: We ate Pizza (1), No Rain (0), We Slept (1).
            
                [ 1 ]
            x = [ 0 ]
                [ 1 ]
        """
        self.stm_size = stm_size     # line format
        """
        The Short-Term Memory (h):
        
            Size: 2 (Happiness, Energy)
            Shape: A vertical line (Vector)
            Data: This is what we want to calculate.(except prevs)

            h = [ ? ]
                [ ? ]
        """
        
        # Initialize weights (keep individual weights in a matrix format)
        """
        The Weight Matrix (W):
            !!!For a simple NN!!!:
            
                Size: 2 x 3 (2 Rows for Outputs, 3 Cols for Inputs)
                Shape: A Grid (Matrix)
                Data: The "recipes" we defined earlier.

                W = [ 1.0  -1.0   0.5 ]  (Row 1: Happiness)
                    [-0.5   0.0   2.0 ]  (Row 2: Energy)
                    
                    
            !!!FOR A LSTM!!!:
            
                W = [ 0.9  0.1  |  1.0  -1.0   0.5 ]  (Row 1: Happiness)
                    [ 0.2  0.8  | -0.5   0.0   2.0 ]  (Row 2: Energy)
                       ^    ^        ^     ^     ^
                     (h_prev)     (current input x)
        """
        
        
        
        # Forget gate
        self.Wf = np.random.randn(stm_size, stm_size + input_size) * 0.01 
        # Shape: (output_size, combined_input_size)
        # Why combined? We stack [stm_prev, x] but matrix treats each POSITION differently via separate column weights
        # Left columns learn "how does PAST affect decision", Right columns learn "how does PRESENT affect decision"
        
        
        self.bf = np.zeros((stm_size, 1))
        # Initialized to zero (network will learn the right baseline during training)
        
        # Input gate  
        self.Wi = np.random.randn(stm_size, stm_size + input_size) * 0.01
        self.bi = np.zeros((stm_size, 1))

        # Candidate(FIRST tanh)
        self.Wc = np.random.randn(stm_size, stm_size + input_size) * 0.01
        self.bc = np.zeros((stm_size, 1))
        
        # Output gate
        self.Wo = np.random.randn(stm_size, stm_size + input_size) * 0.01
        self.bo = np.zeros((stm_size, 1))
        
        
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def forward(self, x, stm_prev, ltm_prev):
        
        """
        Forward pass for one timestep
        
        Parameters:
        -----------
        x : numpy array, shape (input_size, 1)
            Current input
        stm_prev : numpy array, shape (stm_size, 1)  
            Previous short-term memory (hidden state)
        ltm_prev : numpy array, shape (stm_size, 1)
            Previous long-term memory (cell state)
        
        Returns:
        --------
        stm_next : numpy array, shape (stm_size, 1)
            New short-term memory
        ltm_next : numpy array, shape (stm_size, 1)
            New long-term memory
        """
        
        #we care "axb . bxc" so we can calculate dot product
        
        # Step 1: Concatenate previous STM and current input (Standart hidden_state & input merge technique)
        combined_input = np.concatenate([stm_prev, x], axis=0) #matrix merge
        # Shape: (stm_size + input_size, 1)

        # Step 2: Forget gate
        forget_gate = self.sigmoid(np.dot(self.Wf, combined_input) + self.bf) # dot product provide each input get applied by thier own weight so we do not create a "mold" after combination of inputs and make them a process
        # Shape: (stm_size, 1)
        
        # Step 3: Input gate-1 - decides how much new info to add
        input_gate = self.sigmoid(np.dot(self.Wi, combined_input) + self.bi)
        # Shape: (stm_size, 1)

        # Step 4: Input gate-2 (Candidate) - the new potential memory to add
        new_potential_memory = self.tanh(np.dot(self.Wc, combined_input) + self.bc)
        # Shape: (stm_size, 1)
        
        # Step 5:              |Update long-term memory (cell state)
        # Keep some old memory + add some new memory
        ltm_next = forget_gate * ltm_prev + input_gate * new_potential_memory
        
        
        # Step 6: Output gate-1 - decides what to reveal from LTM
        output_gate = self.sigmoid(np.dot(self.Wo, combined_input) + self.bo)
        # Shape: (stm_size, 1)
        
        # Step 7: Output gate-2 |Update short-term memory (hidden state)
        # Filter the LTM through output gate
        stm_next = output_gate * np.tanh(ltm_next)
        
        
        
        # Store for backward
        self.combined_input = combined_input
        self.forget_gate = forget_gate
        self.input_gate = input_gate
        self.new_potential_memory = new_potential_memory
        self.output_gate = output_gate
        self.ltm_prev = ltm_prev
        self.ltm_next = ltm_next

        return stm_next, ltm_next
    
    
    
    
    def forward_sequence(self, x_sequence, stm_init, ltm_init):
        
        stm_outputs=[]
        
        stm_current = stm_init
        ltm_current = ltm_init
        
    
        for x in x_sequence:
            
            x = x.reshape(-1,1) # make 2-d the element in 2-d input matrix (not necessary,it was added to implement a test task however it canceled)
            
            stm_current, ltm_current = self.forward(x, stm_current, ltm_current)
            
            stm_outputs.append(stm_current)

        return stm_outputs, ltm_current


    def backward(self, dstm_next, dltm_next): #from end to start!
        
        
        # dLOSS/dstm * dstm/dvar = dLOSS/dvar          !THE CHAIN RULE!
        """
        Backward pass for one timestep
        
        We calculate gradients (mistake counters) for all weights.
        These tell us how to update weights to reduce error.
        """        
        #we want to declare every gradiant based on dstm because we assume that we already have it.
            
        # Start with gradient from future
        dstm = dstm_next  # dLoss/dstm
                    
                    
        #stm_next = output_gate * tanh(ltm_next)
        #========================================
                
        # Gradient for output_gate
        # dLoss/doutput_gate = dLoss/dstm × dstm/doutput_gate
        # Partner rule: dstm/doutput_gate = tanh(ltm_next)
        do = dstm * np.tanh(self.ltm_next)
        
        # Gradient for ltm_next (PARTIAL - from STM path only)
        # dLoss/dltm = dLoss/dstm × dstm/dltm
        # Chain through tanh: dstm/dltm = output_gate × tanh_derivative
        # Tanh derivative: 1 - tanh²(x)
        dltm = dstm * self.output_gate * (1 - np.tanh(self.ltm_next)**2)
        
        # Add gradient from future timestep
        # ltm_next participates in error through TWO paths:
        #   1. Current STM calculation (calculated above)
        #   2. Future timesteps (passed as dltm_next)
        # Addition rule: add both participation rates
        dltm = dltm + dltm_next  # TOTAL dLoss/dltm
        
        # ================================================================
        # STEP 5: Gradient through LTM calculation (overall gate error rate)
        # Forward was: ltm_next = forget_gate * ltm_prev + input_gate * candidate
        # ================================================================

        # Branch 1: forget_gate × ltm_prev
        # dLoss/dforget_gate = dLoss/dltm × dltm/dforget_gate
        # Partner rule: dltm/dforget_gate = ltm_prev
        df = dltm * self.ltm_prev
        
        # dLoss/dltm_prev = dLoss/dltm × dltm/dltm_prev
        # Partner rule: dltm/dltm_prev = forget_gate
        dltm_prev = dltm * self.forget_gate
        
        # Branch 2: input_gate × candidate
        # dLoss/dinput_gate = dLoss/dltm × dltm/dinput_gate
        # Partner rule: dltm/dinput_gate = candidate
        di = dltm * self.new_potential_memory
        
        # dLoss/dcandidate = dLoss/dltm × dltm/dcandidate
        # Partner rule: dltm/dcandidate = input_gate
        dc = dltm * self.input_gate
        
        # ================================================================
        # STEP 6: Gradient through activation functions (before entering act. func. error rate)
        # ================================================================
        
        # dLoss/doutput_gate_input = dLoss/doutput_gate × doutput_gate/doutput_gate_input
        # Output gate: sigmoid derivative = sigmoid(x) × (1 - sigmoid(x))
        do_input = do * self.output_gate * (1 - self.output_gate)
        
        # dLoss/dinput_gate_input = dLoss/dinput_gate × dinput_gate/dinput_gate_input
        # Input gate: sigmoid derivative
        di_input = di * self.input_gate * (1 - self.input_gate)
        
        # dLoss/dforget_gate_input = dLoss/dforget_gate × dforget_gate/dforget_gate_input
        # Forget gate: sigmoid derivative
        df_input = df * self.forget_gate * (1 - self.forget_gate)
        
        # dLoss/dcandidate_gate_input = dLoss/dcandidate_gate × dcandidate_gate/dcandidate_gate_input
        # Candidate: tanh derivative = 1 - tanh²(x)
        dc_input = dc * (1 - self.new_potential_memory**2)
        
        # ================================================================
        # STEP 7: Gradients for weights and biases
        # Forward was: gate = sigmoid(W @ combined_input + b)
        # ================================================================
        
        #dLoss/dWo = dLoss/doutput_gate_input × doutput_gate_input/dWo
        #output_gate_input = Wo . combined_input + bo
        #output gate weights and bias
        dWo = np.dot(do_input, self.combined_input.T)
        dbo = do_input
        
        #dLoss/dWi = dLoss/dinput_gate_input × dinput_gate_input/dWi
        #input_gate_input = Wi . combined_input + bi
        #input gate weights and bias
        dWi = np.dot(di_input, self.combined_input.T)
        dbi = di_input
        
        #dLoss/dWf = dLoss/dforget_gate_input × dforget_gate_input/dWf
        #forget_gate_input = Wf . combined_input + bf
        # Forget gate weights and bias
        dWf = np.dot(df_input, self.combined_input.T)
        dbf = df_input
        
        #dLoss/dWc = dLoss/dcandidate_gate_input × dcandidate_gate_input/dWc
        #candidate_gate_input = Wc . combined_input + bc
        # Candidate weights and bias
        dWc = np.dot(dc_input, self.combined_input.T)
        dbc = dc_input
        
        # ================================================================
        # STEP 8: Gradient for combined input
        # combined_input affects ALL 4 gates, so ADD all 4 gradients
        # ================================================================
        
        #ANY_gate_input = W . combined_input + b
        dcombined = (np.dot(self.Wo.T, do_input) +
                    np.dot(self.Wi.T, di_input) +
                    np.dot(self.Wf.T, df_input) +
                    np.dot(self.Wc.T, dc_input))
        
        #We sum them up because our input just a whole ONE MATRIX
        
        # ================================================================
        # STEP 9: Split combined input gradient
        # Forward: combined_input = [stm_prev, x]
        # Backward: split it back!
        # ================================================================
        
        dstm_prev = dcombined[:self.stm_size]   # First part: dLoss/dstm_prev
        dx = dcombined[self.stm_size:]           # Second part: dLoss/dx
        
        # ================================================================
        # RETURN all gradients (all are dLoss/d...)
        # ================================================================
        
        return {
            # Weight gradients (for updating during training)
            'dWo': dWo, 'dbo': dbo,  # Output gate
            'dWi': dWi, 'dbi': dbi,  # Input gate
            'dWf': dWf, 'dbf': dbf,  # Forget gate
            'dWc': dWc, 'dbc': dbc,  # Candidate
            
            # Gradients to pass to previous timestep
            'dstm_prev': dstm_prev,  # dLoss/dstm (pass backward)
            'dltm_prev': dltm_prev,  # dLoss/dltm (pass backward)
            
            # Input gradient
            'dx': dx  # dLoss/dx
        }




    def backward_sequence(self, dstm_outputs):
        """
        Backward pass through a sequence (Backpropagation Through Time - BPTT)
        
        Parameters:
        -----------
        dstm_outputs : list of gradients for each timestep's stm output
                    e.g., [dstm_t1, dstm_t2, dstm_t3, dstm_t4, dstm_t5]
                    Each is shape (stm_size, 1)
        
        Returns:
        --------
        accumulated_grads : dictionary with accumulated gradients for all weights
                        These are the TOTAL gradients across all timesteps
        """
        
        
        """
        BACKWARD THROUGH TIME:
        ........
        .....
        ...........
        ...............
        ...........
        
        Time 3:
        Input: dstm=0.5, dltm=0
        Output: dWo=0.03, dstm_prev=0.3, dltm_prev=0.4
        Accumulate: dWo_total = 0.03
        Handover: dstm_next=0.3, dltm_next=0.4 →

        Time 2:
        Input: dstm=0.7, dltm=0.4  ← (received from Time 3)
        Output: dWo=0.08, dstm_prev=0.2, dltm_prev=0.3
        Accumulate: dWo_total = 0.03 + 0.08 = 0.11
        Handover: dstm_next=0.2, dltm_next=0.3 →

        Time 1:
        Input: dstm=0.8, dltm=0.3  ← (received from Time 2)
        Output: dWo=0.05, dstm_prev=0.1, dltm_prev=0.2
        Accumulate: dWo_total = 0.11 + 0.05 = 0.16
        (No more timesteps)

        FINAL: accumulated_grads['dWo'] = 0.16
        """
        # Initialize accumulated gradients (start at zero)
        accumulated_grads = {
            'dWo': np.zeros_like(self.Wo),
            'dbo': np.zeros_like(self.bo),
            'dWi': np.zeros_like(self.Wi),
            'dbi': np.zeros_like(self.bi),
            'dWf': np.zeros_like(self.Wf),
            'dbf': np.zeros_like(self.bf),
            'dWc': np.zeros_like(self.Wc),
            'dbc': np.zeros_like(self.bc)
        }
        
        # Start with zero gradients from "future" (no timestep after the last one)
        dstm_next = np.zeros((self.stm_size, 1))
        dltm_next = np.zeros((self.stm_size, 1))
        
        # Go backwards through time (from last timestep to first)
        for t in reversed(range(len(dstm_outputs))):
            # Add gradient from this timestep's output
            dstm_next = dstm_next + dstm_outputs[t]
            
            # Run backward for THİS timestep
            grads = self.backward(dstm_next, dltm_next)
            
            # Accumulate weight gradients (add them up across timesteps)
            accumulated_grads['dWo'] += grads['dWo']
            accumulated_grads['dbo'] += grads['dbo']
            accumulated_grads['dWi'] += grads['dWi']
            accumulated_grads['dbi'] += grads['dbi']
            accumulated_grads['dWf'] += grads['dWf']
            accumulated_grads['dbf'] += grads['dbf']
            accumulated_grads['dWc'] += grads['dWc']
            accumulated_grads['dbc'] += grads['dbc']
            
            # Pass gradients to previous timestep
            dstm_next = grads['dstm_prev']
            dltm_next = grads['dltm_prev']
        
        return accumulated_grads


    def update_weights(self, grads, learning_rate=0.01):
        
        #There is -= causing of: If gradient is positive → subtract makes weight smaller. If gradient is negative → subtract (negative) = add, makes weight bigger
        
        # Update output gate
        self.Wo -= learning_rate * grads['dWo'] 
        self.bo -= learning_rate * grads['dbo']
        
        # Update input gate
        self.Wi -= learning_rate * grads['dWi']
        self.bi -= learning_rate * grads['dbi']
        
        # Update forget gate
        self.Wf -= learning_rate * grads['dWf']
        self.bf -= learning_rate * grads['dbf']
        
        # Update candidate
        self.Wc -= learning_rate * grads['dWc']
        self.bc -= learning_rate * grads['dbc']

