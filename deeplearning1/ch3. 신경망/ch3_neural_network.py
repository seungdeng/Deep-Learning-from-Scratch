import math as m
import numpy as np
import pandas as pd
import matplotlib.pylab as plt




# =============================================================================
# def step_function2(x):
#     y = x>0
#     return y.astype(np.int64)
# 
# =============================================================================


# =============================================================================
# 
# x = np.array([-1.0,1.0,2.0])
# y = x>0
# print(y)
# 
# y = y.astype(np.int64)
# print(y)
# =============================================================================

# =============================================================================
# def step_function(x):
#     return np.array(x >0,dtype=np.int64)
#     
# x = np.arange(-5.0,5.0,0.1 )
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) #y축의 범위 지정
# plt.show()
# 
# 
# =============================================================================

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0,5.0,0.1 )
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()