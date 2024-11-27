
import numpy as np

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #loss
        self.y = None    #softmax 출력
        self.t = None    #정답 레이블(원-핫 레코딩)
        
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        
        return self.loss
    
    def backward(self,dout =1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)  / batch_size
        
        return dx
        