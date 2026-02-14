# COPY SEQUENCE TEST TASK (f(x) = x)

from lstm import LSTMCell
import numpy as np

all_inputs_size_value = 5
stm_size_value = 128
loop_count = 5000

lstm = LSTMCell(input_size=1, stm_size=stm_size_value)

x_sequence = np.random.rand(all_inputs_size_value,1)

for step in range(loop_count):
    stm_outputs, ltm_current = lstm.forward_sequence(x_sequence,stm_init=np.zeros((stm_size_value,1)),ltm_init=np.zeros((stm_size_value,1)))
    
    stm_array = np.array(stm_outputs)
    stm_reshaped = stm_array.reshape(-1, 128)
    stm_condensed = stm_reshaped.mean(axis=1)
    stm_final = stm_condensed.reshape(-1, 1)
    
    
    #CHECK THİS PART FOR LATER IT DOESNT MAKE THAT SENSE FOR ME FOR NOW!!!!!
    #================================================================#
    
    #(x_sequence - stm_final) ^ 2
    dloss = 2 * (stm_final - x_sequence) / len(x_sequence)
    
    
    dloss_expanded = np.repeat(dloss, stm_size_value, axis=1) / stm_size_value
    
    
    loss_gradient = dloss_expanded.reshape(-1, 1)    
    accumulated_grads = lstm.backward_sequence(loss_gradient)
    #================================================================#


    lstm.update_weights(accumulated_grads,learning_rate=0.005)
    
    print(f"There are {loop_count - step} more loops left to finish!")
        
        
# final test forward
final_outputs, _ = lstm.forward_sequence(x_sequence, np.zeros((stm_size_value,1)), np.zeros((stm_size_value,1)))

final_array = np.array(final_outputs).reshape(-1, 128).mean(axis=1).reshape(-1, 1)

print("\nInput vs Output:")

for i, (inp, out) in enumerate(zip(x_sequence, final_array)):
    print(f"{inp[0]:.1f} -> {out[0]:.4f}")