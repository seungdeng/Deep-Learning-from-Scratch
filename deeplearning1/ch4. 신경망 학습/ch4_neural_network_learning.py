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

# =============================================================================
# import sys,os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# 
# (x_train, t_train), (x_test,t_test) =\
#     load_mnist(normalize=True,one_hot_label=True)
# 
# print(x_train.shape) # (6000,784)
# print(t_train.shape) # (60000,10)
# 
# train_size = x_train.shape[0]
# batch_size =10
# batch_mask = np.random.choice(train_size,batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]
# 
# 
# 
# =============================================================================
# # =============================================================================
# 
# def cross_entropy_error(y,t):
#     if y.ndim ==1:
#         t = t.reshape(1,t.size)
#         y = y.reshape(1,y.size)
#         
#         batch_size = y.shape[0]
#         return -np.sum(t*np.log(y + 1e-7)) / batch_size
# 
# =============================================================================
# =============================================================================
# def cross_entropy_error(y,t):
#     if y.ndim ==1:
#         t = t.reshape(1,t.size)
#         y = y.reshape(1,y.size)
#         
#         batch_size = y.shape[0]
#         return -np.sum(t*np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size
# 
# 
# =============================================================================


# =============================================================================
# #나쁜 구현 예
# def numerical_diff(f,x):
#     h = 1e-50
#     return (f(f+h) - f(x)) / h
# =============================================================================

def numerical_diff(f,x):
    h = 1e-4  #0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    y = 0.01*x**2 + 0.1*x
    return y

# =============================================================================
# import matplotlib.pylab as plt
# 
# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# 
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y)
# plt.show()
# =============================================================================

# =============================================================================
# print(numerical_diff(function_1,5))
# print(numerical_diff(function_1,10))
# 
# =============================================================================
# 접선의 함수를 구하는 함수
# =============================================================================
# def tangent_line(f, x):
#         d = numerical_diff(f, x)
#         # print(d)
#         y = f(x) - d*x
#         return lambda t: d*t + y
# 
# import matplotlib.pylab as plt 
# tf = tangent_line(function_1, 5)
# y2 = tf(x)
# plt.plot(x, y2)
# plt.show()
# =============================================================================

# =============================================================================
# def function_2(x):
#     return x[0]**2 + x[1]**2
#     # or return np.sum(x**2)
# 
# =============================================================================
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0



def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad
# =============================================================================
# 
# print(numerical_gradient(function_2, np.array([3.0,4.0])))
# print(numerical_gradient(function_2, np.array([0.0,2.0])))
# print(numerical_gradient(function_2, np.array([3.0,0.0])))
# 
# =============================================================================

def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2
    # or return np.sum(x**2)
    
# =============================================================================
# init_x = np.array([-3.0,4.0])
# print(gradient_descent(function_2, init_x=init_x, lr=0.1,step_num=100))
# 
# 
# 
# # 학습률이 너무 큼
# init_x = np.array([-3.0, 4.0])
# x= gradient_descent(function_2, init_x, lr=10.0)
# print(x)  # [ -2.58983747e+13  -1.29524862e+12] 발산함
# 
# # 학습률이 너무 작음
# init_x = np.array([-3.0, 4.0])
# x= gradient_descent(function_2, init_x, lr=1e-10)
# print(x)  # [-2.99999994  3.99999992] 거의 변화 없음
# 
# 
# =============================================================================



import sys
import os
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# 4.4.2 신경망에서의 기울기
class simpleNet:
    """docstring for simpleNet"""
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)  # 가중치 매개변수(랜덤)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))  # 최댓값의 인덱스

t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))

# =============================================================================
# def f(W):
#     return net.loss(x,t)
# 
# dW = numerical_gradient(f,net.W)
# print(dW)
# =============================================================================

f = lambda w:net.loss(x, t)
dW = numerical_gradient(f,net.W)
print(dW)














