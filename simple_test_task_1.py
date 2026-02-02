# COPY SEQUENCE TEST TASK
from lstm import LSTMCell
import numpy as np

lstm = LSTMCell(input_size=1, stm_size=1)

x_sequence = np.arange(10) # np.arange(10).reshape(-1,1) makes EACH ELEMENT 2-D NOT END VARİABLE ORDER (2-d elements in 1-d array)
x_sequence = x_sequence.reshape(-1,1) #makes final variable 2-D (1-d elements in 2-d array)

stm_outputs, ltm_current = lstm.forward_sequence(x_sequence,stm_init=[[0]],ltm_init=[[0]])

stm_outputs = np.array(stm_outputs).reshape(-1, 1) #make it 2-d shape from a list to be able to make operation

asr = (x_sequence - stm_outputs) ** 2 #find "All Squared Residuals"

ssr = asr.sum() #find "Sum of Squared Residuals"s

accumulated_grads = lstm.backward_sequence(asr)

lstm.update_weights(accumulated_grads,learning_rate=0.01)

