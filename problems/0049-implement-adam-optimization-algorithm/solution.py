import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here

    x = np.array(x0, dtype = "float")
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    # First moment
    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_bias =  m / (1-beta1**t)
        v_bias  = v / (1-beta2**t)
        x = x - learning_rate * m_bias / (np.sqrt(v_bias) + epsilon)
    return x

        
  