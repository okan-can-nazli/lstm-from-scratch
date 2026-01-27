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
        # Why combined? We stack [h_prev, x] but matrix treats each POSITION differently via separate column weights
        # Left columns learn "how does PAST affect decision", Right columns learn "how does PRESENT affect decision"
        
        self.bf = np.zeros((stm_size, 1))
        # Bias: one value per OUTPUT, shifts the result before activation
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
    

    

    
