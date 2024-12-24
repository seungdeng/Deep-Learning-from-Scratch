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