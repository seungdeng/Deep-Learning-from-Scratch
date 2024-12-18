import numpy as np

c = np.array([1, 0, 0, 0, 0, 0, 0]) #입력
W = np.random.randn(7, 3)           #가중치
h = np.matmul(c,W)                  #중간 노드
print(h)                            #[[-0.70012195 0.25204755 -0.79774592]]