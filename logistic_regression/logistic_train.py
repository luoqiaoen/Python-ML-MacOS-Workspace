import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_binary_data

X, Y = get_binary_data()
X, Y = shuffle(X,Y)

Xtrain = X[:-100]#up to the last 101
Ytrain = Y[:-100]
Xtest = X[-100:]#last 100
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / (1+np.exp(-a))

def forward(X,W,b):
    return sigmoid(X.dot(W)+b)

def classification_rate(Y,P):
    return np.mean(Y==P)

def cross_entropy(T,pY):
    return -np.mean(T*np.log(pY)+(1-T)*np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in xrange(10000):
    pYtrain = forward(Xtrain,W,b)
    pYtest = forward(Xtest,W,b)

    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest,pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()

    if i % 1000 == 0:
        print i,ctrain,ctest

print "Final train classficiation rate:", classification_rate(Ytrain, np.round(pYtrain))
print "Final test classification rate:", classification_rate(Ytest, np.round(pYtest))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label ='test cost')

plt.legend([legend1, legend2])
plt.show()
