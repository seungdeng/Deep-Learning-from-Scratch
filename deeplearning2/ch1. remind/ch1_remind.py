import numpy as np

x = np.array([1,2,3])
print(x.__class__) # <class 'numpy.ndarray'>
x.shape # (3,)
x.ndim # 1

W = np.array([[1,2,3],[4,5,6]])
W.shape #(2,3)
W.ndim # 2