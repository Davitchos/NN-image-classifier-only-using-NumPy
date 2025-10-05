# Building a Neural Network from Scratch

Credit: the contents of this project are heavily inspired by [this video tutorial](https://www.youtube.com/watch?v=w8yWXqWQYmU).  
The network structure and training/test data set follow this video, but several contributions have been made:
- More efficient parameter initialization was adapted
  - He for layer with ReLU activation
  - Xavier for layer with softmax activation
  - Resulted in more stable and accurate learning increasing accuracy by several percent
- Complete visualizer for whole network with weights of each node and indication of outcomes

---

## Task

Identify hand-drawn digits using a simple neural network trained on the **MNIST dataset** (60,000 training and 10,000 test images of digits 0–9, each 28×28 pixels).

---

## Mathematical & programming techniques

- **Linear algebra & vectorization:** forward and backward propagation fully vectorized with NumPy.  
- **Activation functions:** ReLU for hidden layers, softmax for output probabilities.  
- **Initialization strategies:** He initialization for ReLU, Xavier/Glorot for output layer.  
- **Gradient descent:** parameter updates with backpropagated derivatives.  
- **One-hot encoding:** efficient label representation for multiclass classification.  
- **Visualization:** matplotlib for inspecting samples and predictions.  
- **Parallelizable training (optional):** code is simple enough to extend to multiprocessing or GPU frameworks.

---

## Contents

- Data preparation (MNIST CSV → NumPy arrays, scaling, shuffle, dev/test split).  
- Fully connected neural net (1 hidden layer) from scratch.  
- Forward pass: $Z = WX + b$, $A = \mathrm{ReLU}(Z)$, softmax for output.  
- Backpropagation: computing gradients w.r.t. $W$ and $b$.  
- Training loop with accuracy tracking every 50 iterations.  
- Testing on unseen dev set and visualization of predictions.

---

## Quick use

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=1000, alpha=0.05)
test_prediction(0, W1, b1, W2, b2)
```

## Possible improvements (to do)
- Add regularization (L2 / dropout) / experiment with different learning rates
- Increase complexity -> add more layers
- Port to GPU with PyTorch / TensorFlow if needed
