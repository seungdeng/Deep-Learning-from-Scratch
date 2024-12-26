# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:11:48 2024

Created by SeungKeon Lee
"""

import sys
import numpy as np
try:
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False

sys.path.append('..')
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        if GPU_ENABLED:
            W_in = cp.array(W_in)
            W_out = cp.array(W_out)

        # Layer 생성
        self.in_layers = []
        for _ in range(2 * window_size):
            layer = Embedding(W_in)  # Embedding 계층 사용
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        if GPU_ENABLED and isinstance(contexts, np.ndarray):
            contexts = cp.array(contexts)
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward()
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
