import numpy as np

x = np.array([1,2,3])
print(x.__class__) # <class 'numpy.ndarray'>
x.shape # (3,)
x.ndim # 1

W = np.array([[1,2,3],[4,5,6]])
W.shape #(2,3)
W.ndim # 2

A = np.array([[1,2],[3,4]])

A * 10  # ([[10,20],[30,40]])


A = np.array([[1,2],[3,4]])
b = np.array([10,20])

A * b #array([[10,40],[30,80]])


#벡터의 내적
a = np.array([1,2,3])
b = np.array([4,5,6])
np.dot(a,b) # 벡터의 내적 32

#행렬의 곱
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
np.matmul(A,B) #행렬의 곱 array([[19,22],[43,50]])


W1 = np.random.randn(2,4)   #가중치
b1 = np.random.randn(4)     #편향
x = np.random.randn(10, 2)  #입력
h = np.matmul(x, W1) + b1

def sigmoid(x):
    return 1/(1 + np.exp(-x))


# =============================================================================
# class TwoLayerNet:
#     def __init__(self,input_size,hidden_size,output_size):
#         I,H,P = input_size,hidden_size,output_size
#         
#         #weight과 bias 초기화
#         W1 = np.random.randn(I,H)
#         b1 = np.random.randn(H)
#         W2 = np.random.randn(H,O)
#         b2 = np.random.randn(O)
#         
#         #layer 생성
#         self.layers = [
#             Affine(W1,b1),
#             Sigmoid(),
#             Affine(W2,b2
#             ]
#                  
#         #모든 weight를 리스트에 모은다.
#         self.params = []
#         for layer in self.layers:
#             self.params += layers.params
#             
#     def predict(self,x):
#         for layer in self.layers:
#             x = layer.forward(x)
#             return x
# =============================================================================


D,N = 8,7
x = np.random.randn(1, D)   #입력
y = np.repeat(x,N,axis = 0) #순전파

dy = np.random.randn(N,D)   # 무작위 기울기
dx = np.sum(dy,axis=0,keepdims=True) #역전파