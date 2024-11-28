# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:33:00 2024

@author: SGLEE
"""

import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
            
# =============================================================================
# network = TwoLayerNet(...)
# optimizer = SGD()
# 
# for i  in range(10000):
#     ...
#     x_batch, t_batch = get_mini_batch(...) #mini batch
#     grads = network.gradient(x_batch,t_batch)
#     params = network.params
#     optimizer.update(params,grads)
# =============================================================================

class Momentum:
    def __init__(self,lr=0.01,momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key,val in parmas.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]