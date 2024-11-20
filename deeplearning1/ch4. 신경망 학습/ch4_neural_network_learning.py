import numpy as np

def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

# =============================================================================
# # 정답은 2
# t = [0,0,1,0,0,0,0,0,0,0]
# 
# # 2일 확률이 제일높다
# y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
# 
# print(sum_squares_error(np.array(y),np.array(t)))
# 
# # 7일 확률이 제일 높다
# y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
# 
# print(sum_squares_error(np.array(y),np.array(t)))
# =============================================================================

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# =============================================================================
# # 정답은 2
# t = [0,0,1,0,0,0,0,0,0,0]
# 
# # 2일 확률이 제일높다
# y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
# 
# print(cross_entropy_error(np.array(y),np.array(t)))
# 
# # 7일 확률이 제일 높다
# y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
# 
# print(cross_entropy_error(np.array(y),np.array(t)))
# =============================================================================

import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test,t_test) =\
    load_mnist(normalize=True,one_hot_label=True)

print(x_train.shape) # (6000,784)
print(t_train.shape) # (60000,10)