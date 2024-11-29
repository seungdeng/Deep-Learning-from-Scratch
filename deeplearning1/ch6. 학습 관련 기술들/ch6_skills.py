# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:33:00 2024

@author: SGLEE
"""

import numpy as np
import matplotlib.pyplot as plt


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
# =============================================================================
# 
# class Momentum:
#     def __init__(self,lr=0.01,momentum = 0.9):
#         self.lr = lr
#         self.momentum = momentum
#         self.v = None
#         
#     def update(self,params,grads):
#         if self.v is None:
#             self.v = {}
#             for key,val in parmas.items():
#                 self.v[key] = np.zeros_like(val)
#             
#         for key in params.keys():
#             self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
#             params[key] += self.v[key]
# =============================================================================

class AdaGrad:
    def __init__(self, lr= 0.01):
        self.lr = lr
        self.h = None
        
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))
            
            
            
            
def sigmoid(x):
    return 1/ (1+np.exp(-x))

x = np.random.randn(1000,100)   #1000개의 데이터
node_num = 100                  #각 은닉층의 노드(뉴런) 수
hidden_layer_size =5            #은닉층 5개
activations = {}                #이곳에 활성화값 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
        w = np.random.randn(node_num,node_num) * 1
        a = np.dot(x,w)
        z = sigmoid(a)
        activations[i] = z
        
for i,a in activations.items():
    plt.subplot(1, len(activations), i +1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(),30,range(30,1))
    
plt.show()
        

# =============================================================================
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True)
# x_train = x_train[:300]
# t_train = t_train[:300]
# network = MultiLayerNet(input_size = 784, hidden_size_list = [100,100,100,100,100,100],output_size = 10)
# optimizer = SGD(lr = 0.01)
# =============================================================================


class Dropout:
    def __init__(self,dropout_ratio =0.5):
        self.dropout_ration = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
            
    def backward(self,dout):
        return dout * self.mask
        


(x_train, t_train), (x_test, t_test) = load_mnist()

x_train, t_train = shuffle_dataset(x_train, t_train)

validation_rate = 0.20
validation_num = int(x_+train.shape[0] * validation_rate)

x_val = x_train[validation_num:]
t_val = t_train[validation_num:]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]