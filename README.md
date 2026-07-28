# Recurrent Neural Networks from Scratch (RNN, LSTM, GRU)

Pure NumPy implementations of RNN, LSTM, and GRU cells — no PyTorch, no TensorFlow, no autograd. Every forward pass, backward pass, and gradient update is written and derived by hand, including full BPTT (Backpropagation Through Time).

This repo also includes a **Caputo fractional derivative** extension for each architecture, replacing standard gate derivatives with a fractional-order version based on the Caputo definition. This lets each gate's gradient incorporate a memory of past derivatives, rather than depending only on the current timestep.

## Structure

```
Standart RNN Framework/
    rnn.py     # RNN cell
    lstm.py    # LSTM cell
    gru.py     # GRU cell

Fractional RNN Framework/ folder of cells with Caputo fractional derivatives
    CaputoRNN.py    # RNN cell 
    CaputoLstm.py   # LSTM cell
    CaputoGRU.py    # GRU cell 
```

Each folder mirrors the same three architectures — the "Fractional" versions are drop-in variants of the "Standart" ones, with an added `sigma` parameter controlling the fractional order (`sigma=1.0` reduces to the standard derivative).

## Why fractional derivatives?

The Caputo fractional derivative generalizes ordinary differentiation to non-integer orders:

```
D^sigma f(t) = (1/Γ(1-sigma)) * Σ[k=0 to t] f'(k) / (t-k)^sigma
```

Instead of a gate's gradient depending only on its current derivative, it becomes a *weighted sum of all past derivatives* in the sequence, with weights that decay based on `sigma`. Lower `sigma` values give past timesteps more influence — a way of building long-range memory directly into the gradient computation itself, rather than relying solely on the architecture's gating mechanism.

## Usage

```python
from lstm import LSTMCell
import numpy as np

cell = LSTMCell(input_size=1, stm_size=32)

x_sequence = [np.array([i * 0.1]) for i in range(50)]
stm_init = np.zeros((32, 1))
ltm_init = np.zeros((32, 1))

stm_outputs, caches = cell.forward_sequence(x_sequence, stm_init, ltm_init)

# dy_preds: list of output gradients, one per timestep
grads = cell.backward_sequence(dy_preds, stm_outputs, caches)
cell.update_weights(grads, learning_rate=0.01)
```

Fractional variants take an extra `sigma` argument in their backward pass:

```python
from CaputoLstm import LSTMCell as CaputoLSTMCell

cell = CaputoLSTMCell(input_size=1, stm_size=32)
grads = cell.backward_sequence(dy_preds, stm_outputs, caches, sigma=0.7)
cell.update_weights(grads, learning_rate=0.01)
```

## Notes

- All gate derivatives assume the **already-activated** value as input (e.g. `sigmoid_derivative(x) = x*(1-x)` expects `x` to already be a sigmoid output), matching the convention used throughout.
- Gradient clipping is recommended for longer sequences to avoid exploding gradients through BPTT.
- These implementations prioritize clarity and mathematical correctness over performance — they're built for understanding the mechanics, not for production training speed.