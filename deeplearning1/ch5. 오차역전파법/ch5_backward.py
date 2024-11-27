class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self,dout):
        dx = dout * self.y #x와 y를 바꾼다.
        dy = dout * self.x
        
        return dx,dy
        

# =============================================================================
# apple = 100
# apple_num = 2
# tax = 1.1
# 
# #계층
# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()
# 
# #순전파
# apple_price = mul_apple_layer.forward(apple,apple_num)
# price = mul_tax_layer.forward(apple_price,tax)
# print(price)  #220
# 
# #역전파 
# dprice = 1
# dapple_price,dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# print(dapple,dapple_num,dtax) # 2.2 110 220
# =============================================================================


class AddLayer:
    def __init__(self):
        pass
    
    def forward(self,x,y):
        out = x+y
        return out
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy
# =============================================================================
#     
# apple = 100
# apple_num = 2
# orange = 150
# orange_num = 3
# tax = 1.1
# 
# mul_apple_layer = MulLayer()            
# mul_orange_layer = MulLayer()           
# add_apple_orange_layer = AddLayer()     
# mul_tax_layer = MulLayer()
# 
# apple_price = mul_apple_layer.forward(apple, apple_num)                 #1
# orange_price = mul_orange_layer.forward(orange, orange_num)             #2
# all_price = add_apple_orange_layer.forward(apple_price, orange_price)   #3
# price = mul_tax_layer.forward(all_price, tax)                           #4
# 
# dprice = 1
# dall_price, dtax = mul_tax_layer.backward(dprice)                           #4
# dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)   #3
# dorange, dorange_num = mul_orange_layer.backward(dorange_price)             #2
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)                 #1
# 
# print(price)    #715
# print(dapple_num,dapple,dorange,dorange_num,dtax)   # 110 2.2 3.3 165 650
# =============================================================================

import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self,x):
        self.mask = (x <=0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
    
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self,dout):
        dx = dout * (1.0 - self.out ) * self.out
        
        return dx
    
    
    
    
# =============================================================================
# X_dot_W =  np.array([[0,0,0],[10,10,10]])
# B = np.array([1,2,3])
# 
# print(X_dot_W)
# print(X_dot_W+B)
# =============================================================================

    

dY = np.array([[1,2,3],[4,5,6]])
print(dY)

dB = np.sum(dY,axis=0)
print(dB)
    
    
    
    
    