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
        
        # Step 1: Concatenate previous STM and current input
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
        
            
        # Start with gradient from future
        dstm = dstm_next  # dLoss/dstm
                    
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
        # Each gate has sigmoid or tanh activation
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













# Test code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING LSTM FORWARD PASS")
    print("="*60 + "\n")
    
    # Create LSTM
    input_size = 3
    stm_size = 4
    lstm = LSTMCell(input_size, stm_size)
    print(f"✓ Created LSTM (input_size={input_size}, stm_size={stm_size})\n")
    
    # Create test inputs
    x = np.random.randn(input_size, 1) * 0.5
    stm_prev = np.random.randn(stm_size, 1) * 0.5  
    ltm_prev = np.random.randn(stm_size, 1) * 0.5
    
    print("Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  stm_prev: {stm_prev.shape}")
    print(f"  ltm_prev: {ltm_prev.shape}\n")
    
    # Forward pass TESTING
    stm_next, ltm_next = lstm.forward(x, stm_prev, ltm_prev)
    
    print("="*60)
    print("✅ FORWARD PASS SUCCESSFUL!")
    print("="*60 + "\n")
    
    print("Output shapes:")
    print(f"  stm_next: {stm_next.shape}")
    print(f"  ltm_next: {ltm_next.shape}\n")
    
    print("Output values:")
    print(f"  stm_next range: [{stm_next.min():.4f}, {stm_next.max():.4f}]")
    print(f"  ltm_next range: [{ltm_next.min():.4f}, {ltm_next.max():.4f}]")
    
    print("\nstm_next:")
    print(stm_next)
    print("\nltm_next:")
    print(ltm_next)
    
    
    
    
    #Forward sequence TESTING
    print("\n" + "="*60)
    print("TESTING SEQUENCE PROCESSING")
    print("="*60 + "\n")
    
    # Create a sequence of 5 timesteps
    sequence_length = 5
    x_sequence = [np.random.randn(input_size, 1) * 0.5 for _ in range(sequence_length)]
    
    print(f"Created sequence of {sequence_length} timesteps")
    print(f"Each input shape: {x_sequence[0].shape}\n")
    
    # Initial states (zeros)
    stm_init = np.zeros((stm_size, 1))
    ltm_init = np.zeros((stm_size, 1))
    
    # Process sequence
    stm_outputs, ltm_final = lstm.forward_sequence(x_sequence, stm_init, ltm_init)
    
    print("="*60)
    print("✅ SEQUENCE PROCESSING SUCCESSFUL!")
    print("="*60 + "\n")
    
    print(f"Processed {len(stm_outputs)} timesteps")
    print(f"Each STM output shape: {stm_outputs[0].shape}")
    print(f"Final LTM shape: {ltm_final.shape}\n")
    
    print("STM at each timestep:")
    for t, stm in enumerate(stm_outputs):
        print(f"  t={t+1}: range [{stm.min():.4f}, {stm.max():.4f}]")
        
        
        
    # Test backward pass
print("\n" + "="*60)
print("TESTING BACKWARD PASS")
print("="*60 + "\n")

# First run forward to cache values
x = np.random.randn(input_size, 1) * 0.5
stm_prev = np.random.randn(stm_size, 1) * 0.5
ltm_prev = np.random.randn(stm_size, 1) * 0.5

stm_next, ltm_next = lstm.forward(x, stm_prev, ltm_prev)
print("✓ Forward pass completed (cached values for backward)\n")

# Create dummy gradients from "future"
dstm_next = np.random.randn(stm_size, 1) * 0.1
dltm_next = np.random.randn(stm_size, 1) * 0.1

print("Input gradients:")
print(f"  dstm_next: {dstm_next.shape}")
print(f"  dltm_next: {dltm_next.shape}\n")

# Run backward
grads = lstm.backward(dstm_next, dltm_next)

print("="*60)
print("✅ BACKWARD PASS SUCCESSFUL!")
print("="*60 + "\n")

print("Weight gradient shapes:")
print(f"  dWf: {grads['dWf'].shape} (should be {lstm.Wf.shape})")
print(f"  dWi: {grads['dWi'].shape} (should be {lstm.Wi.shape})")
print(f"  dWc: {grads['dWc'].shape} (should be {lstm.Wc.shape})")
print(f"  dWo: {grads['dWo'].shape} (should be {lstm.Wo.shape})\n")

print("Bias gradient shapes:")
print(f"  dbf: {grads['dbf'].shape} (should be {lstm.bf.shape})")
print(f"  dbi: {grads['dbi'].shape} (should be {lstm.bi.shape})")
print(f"  dbc: {grads['dbc'].shape} (should be {lstm.bc.shape})")
print(f"  dbo: {grads['dbo'].shape} (should be {lstm.bo.shape})\n")

print("Gradients to pass backward:")
print(f"  dstm_prev: {grads['dstm_prev'].shape}")
print(f"  dltm_prev: {grads['dltm_prev'].shape}")
print(f"  dx: {grads['dx'].shape}\n")

print("Sample gradient values:")
print(f"  dWf range: [{grads['dWf'].min():.6f}, {grads['dWf'].max():.6f}]")
print(f"  dstm_prev range: [{grads['dstm_prev'].min():.6f}, {grads['dstm_prev'].max():.6f}]")
