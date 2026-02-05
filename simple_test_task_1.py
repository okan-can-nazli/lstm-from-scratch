# COPY SEQUENCE TEST TASK
from lstm import LSTMCell
import numpy as np

lstm = LSTMCell(input_size=1, stm_size=1)

x_sequence = np.random.rand(100,1)

for step in range(5000):
    stm_outputs, ltm_current = lstm.forward_sequence(x_sequence,stm_init=np.zeros((1,1)),ltm_init=np.zeros((1,1)))
    stm_outputs = np.array(stm_outputs).reshape(-1, 1) #make it 2-d shape from a list to be able to make operation for all stm values

    stm_formatted = stm_outputs.reshape(100, 1)

    loss = ((x_sequence - stm_formatted) ** 2).mean() #find "All Squared Residuals" in a matrix shaped (100,1)


    #CHECK THİS PART FOR LATER IT DOESNT MAKE THAT SENSE FOR ME FOR NOW!!!!!
    #================================================================#
    loss_gradient = 2 * (stm_formatted - x_sequence) / len(x_sequence)
    
    loss_gradient = loss_gradient.reshape(100, 1, 1)
    
    accumulated_grads = lstm.backward_sequence(loss_gradient)
    #================================================================#





    lstm.update_weights(accumulated_grads,learning_rate=0.01)


        
        
        
# Final Test Forward
final_outputs, _ = lstm.forward_sequence(x_sequence, np.zeros((1,1)), np.zeros((1,1)))
final_outputs = np.array(final_outputs).reshape(-1, 1)

print("\nInput vs Output:")
for i, (inp, out) in enumerate(zip(x_sequence, final_outputs)):
    print(f"{inp[0]:.1f} -> {out[0]:.4f}")