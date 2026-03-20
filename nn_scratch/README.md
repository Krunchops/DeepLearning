# nn_from scratch

Built this after watching Andrej Karpathy's Neural Networks: Zero to Hero. Watched it once, then rebuilt everything from memory in a fresh notebook — no peeking.

Scalar-valued autograd engine with a neural net library on top. No PyTorch, no NumPy. Pure Python.

## What it is

A `Value` class that wraps a single float. Every operation builds a computation graph on the fly. Call `.backward()` and it walks the graph in reverse, computing gradients for every value that contributed to the output via the chain rule.

On top of that — `Neuron`, `Layer`, `MLP`. Same abstractions as PyTorch, but ~150 lines so you can actually see what's going on.

## How backprop works here

Each `Value` tracks:
- `data` — the scalar
- `grad` — gradient w.r.t the loss (filled by `.backward()`)
- `_backward` — closure that applies chain rule for this op
- `_prev` — parent nodes in the graph

`.backward()` does a topological sort, sets `loss.grad = 1.0`, then walks backwards calling `_backward()` at each node. By the end every parameter has a `.grad`.
```
forward pass  →  builds the graph
.backward()   →  topo sort → chain rule at every node
```

## Training loop
```python
for step in range(100):
    ypred = [model(x) for x in xs]
    loss  = sum((yp - Value(yt))**2 for yp, yt in zip(ypred, ys))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad
```

## PyTorch comparison

Trained the same network in PyTorch — identical weights, same lr, same manual SGD. Loss curves overlap to under 1e-5 across 100 steps. That was the real check — if the gradients were off, the losses would diverge.
