import numpy as np

x = np.random.rand(10,1,28,28) #무작위 데이터 생성
x.shape # (10,1,28,28)

#첫번째, 두번째 데이터에 접근할 때
x[0].shape # (1,28,28)
x[1].shape # (1,28,28)

#첫번째 데이터의 첫 채널의 공간 데이터에 접근할 때
x[0,0] # or x[0][0]

# =============================================================================
# im2col(input_data, filter_h, filter_w, stride = 1, pad = 0)
# =============================================================================

import sys,os
sys.path.append(os.pardir)
from common.util import im2col 

x1 = np.random.rand(1,3,7,7) #(데이터 수, 채널수, 높이, 너비)
col1 = im2col(x1,5,5,stride = 1, pad = 0)
print(col1.shape) #(9,75)

x2 = np.random.rand(10,3,7,7) #데이터 10개
col2 = im2col(x2,5,5,stride = 1,pad = 0)
print(col2.shape) #(90,75)


class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self,x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN,-1).T
        out = np.dot(col,col_W) + self.b
        
        out = out.reshape(N, out_h,out_w,-1).transpose(0,3,1,2)
        
        return out
    
class Pooling:
    def __init__(self,pool_h,pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1 + (H-self.pool_h) / self.stride)
        out_w = int(1 + (W-self.pool_w) / self.stride)
        
        #전개(1)
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)
        
        #최댓값(2)
        out = np.max(col,axis=1)
        
        #성형(3)
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        
        return out
        
class SimpleConvNet:
    def __init__(self, input_dim = (1,28,28),
                 conv_param={'filter_num':30, 'filter_size':5,
                             'pad':0, 'stride':1},
                 hidden_size =100, output_size=10, weight_int_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        
        conv_output_size(input_size - filter_size + 2*filter_pad) /\
            filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) *
                               (conv_output_size/2))
        
        
        
        
self.param = {}
self.params['W1'] = weight_int_std *\
    np.random.rand(filter_num, input_din[0], filter_size, filter_size)
self.params['b1'] = np.zeros(filter_num)
self.params['W2'] = weight_int_std *\
    np.random.rand(pool_output_size, hidden_size)
self.params['b2'] = np.zeros(hidden_size)
self.params['W3'] = weight_int_std *\
    np.random.rand(hidden_size, output_size)
self.params['b3'] = np.zeros(output_size)






self.layers = OrderedDict()
self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                   conv_param['stride'],conv_param['pad'])
self.layers['Relu1'] = Relu()
self.layers['Pool1'] = Pooling(pool_h=2,pool_w=2,stride=2)
self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])
self.layers['Relu2'] = Relu()
self.layers['Affine2'] = Affine(self.params['W3'],
                                self.params['b3'])
self.last_layer = SoftmaxWithLoss()


def predict(self,x):
    for layer in self.layers.values():
        x = layer.forward(x)
        return x
    
    def lostt(self,x,t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)