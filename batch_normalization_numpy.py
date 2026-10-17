#!/usr/bin/env python
# File Name: batch_normalization_numpy.py
#
# Date Created: Oct 14,2026
#
# Last Modified: Wed Oct 14 22:13:55 2026
#
# Author: samuelo
#
# Description:	
#
##################################################################


import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        """
        Initializes the BatchNorm layer.
        
        Parameters:
        - num_features: Number of channels/features in the input.
        - eps: Small constant for numerical stability (prevents division by zero).
        - momentum: Momentum factor used for computing running mean and variance.
        """
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters (initialized to 1 and 0 respectively)
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # Running statistics used during inference/testing
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backpropagation
        self.cache = None

    def forward(self, x, training=True):
        """
        Forward Pass for Batch Normalization.
        x shape: (batch_size, num_features)
        """
        if training:
            # 1. Calculate mini-batch mean and variance
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # 2. Normalize the input
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            
            # 3. Scale and shift
            out = self.gamma * x_hat + self.beta
            
            # 4. Update running statistics for test time
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Save variables needed for the backward pass
            self.cache = (x, x_hat, mean, var)
        else:
            # During evaluation, use the tracked running statistics
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
            
        return out

    def backward(self, dout):
        """
        Backward Pass for Batch Normalization.
        dout shape: (batch_size, num_features) - Gradient of loss with respect to layer output
        """
        # Retrieve cached variables from the forward pass
        x, x_hat, mean, var = self.cache
        N = x.shape[0]  # Batch size
        
        # 1. Gradients with respect to learnable parameters gamma and beta
        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # 2. Gradient with respect to x_hat
        dx_hat = dout * self.gamma
        
        # 3. Standard backpropagation sequence through the BatchNorm computational graph
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=0, keepdims=True)
        
        dmean = np.sum(dx_hat * -1.0 / np.sqrt(var + self.eps), axis=0, keepdims=True) + \
                dvar * np.sum(-2.0 * (x - mean), axis=0, keepdims=True) / N
                
        dx = dx_hat / np.sqrt(var + self.eps) + \
             dvar * 2.0 * (x - mean) / N + \
             dmean / N
             
        return dx, dgamma, dbeta




############## Verification ##############
# Create random input: 4 samples, 3 features
np.random.seed(42)
X_batch = np.random.randn(4, 3) * 10 + 5

bn_layer = BatchNorm(num_features=3)

# 1. Forward Pass (Training)
output_train = bn_layer.forward(X_batch, training=True)
print("--- Training Output ---")
print(output_train)
print("\nMean of output (should be close to 0):\n", np.mean(output_train, axis=0))
print("Variance of output (should be close to 1):\n", np.var(output_train, axis=0))

# 2. Backward Pass
# Pretend the gradient coming back from the next layer is all ones
dout = np.ones_like(X_batch)
dx, dgamma, dbeta = bn_layer.backward(dout)
print("\n--- Backward Gradients ---")
print("Gradient w.r.t input (dx) shape:", dx.shape)
print("Gradient w.r.t gamma (dgamma):", dgamma)

