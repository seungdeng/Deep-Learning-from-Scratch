import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train),(x_test,t_test)  = \
    load_mnist(normalize=True, one_hot_label= True)
    
train_loss_list = []
train_acc_list = []
test_acc_list = []


#하이퍼파라미터
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니배치 크기
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


iter_per_epoch = max(train_size / batch_size , 1)
# =============================================================================
# 
# x = np.arange(0,100,1)
# y = np.arange(-0.1,1.1,0.05 )
# =============================================================================

for i in range(iters_num):
    #미니배치 획득
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #기울기 계산
    grad = network.gradient(x_batch, t_batch)
    # numerical_gradient 말고 gradient가 성능개선판
    
    #매개변수 갱신
    for key in ('W1', 'b1','W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    
    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    #1에폭당 정확도 계산
    if i % iter_per_epoch ==0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc : " + str(train_acc)+", "+str(test_acc))
        
# 정확도 그래프 출력
epochs = np.arange(len(train_acc_list))  # 에폭 범위 생성
plt.plot(epochs, train_acc_list, label="Train Accuracy")
plt.plot(epochs, test_acc_list, label="Test Accuracy", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Test Accuracy")
plt.show()
        
# =============================================================================
#         y1 = test_acc
#         y2 = test_acc
#         plt.plot(x,y1,label="train acc")
#         plt.plot(x,y2,label="test acc",linestyle="--")
#         plt.xlabel("epochs")
#         plt.ylabel("accuracy")
#         plt.legend()
#         plt.show()
# =============================================================================
        
#print(train_loss_list)