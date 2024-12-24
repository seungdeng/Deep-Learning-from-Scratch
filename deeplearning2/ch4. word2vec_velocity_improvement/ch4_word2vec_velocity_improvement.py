# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:09:10 2024

Created by SeungKeon Lee
"""

import numpy as np
W = np.arange(21).reshape(7, 3)
print(W)
print(W[2])
print(W[5])

idx = np.array([1,0,3,0])
print(W[idx])

class Embedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
        def forward(self,idx):
            W, = self.params
            self.idx = idx
            out = W[idx]
            return out
# =============================================================================
#         
#         def backward(self,dout):
#             dW, =  self.grads
#             dW[...] = 0
#             dW[self.idx] = dout
#             return None
# =============================================================================


        def backward(self,dout):
            dW, =  self.grads
            dW[...] = 0
      
            for i, word_id in enumerate(self.idx):
                dW[word_id] += dout[i]
                
            #혹은
            # np.add.at(dW,self.idx,dout)
            return None
        
        
#0에서 9까지의 숫자 중 하나를 무작위로 샘플링        
np.random.choice(10)
np.random.choice(10)

#words에서 하나만 무작위로 샘플링
words = ['you','say','goodbye','I','hello','.']
np.random.choice(words)

#5개만 무작위로 샘플링(중복 있음)
np.random.choice(words,size = 5)

# '' (중복 없음)
np.random.choice(words, size = 5, replace = False)

#확률분포에 따라 샘플링
p = [0.5,0.1,0.05,0.2,0.05,0.1]
np.random.choice(words, p = p)

p =[0.7,0.29,0.01]
new_p = np.power(p,0.75)
new_p /= np.sum(new_p)
print(new_p)


from negative_sampling_layer import UnigramSampler
corpus = np.array([0,1,2,3,4,1,2,3])
power = 0.75
sample_size = 2
sampler = UnigramSampler(corpus,power,sample_size)
target = np.array([1,3,0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)


import sys
sys.path.append('..')
from common.util import analogy
analogy('man', 'king', 'woman', word_to_id, id_to_word, word_vecs, top = 5)
