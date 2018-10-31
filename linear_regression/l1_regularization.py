import numpy as np
import matplotlib.pyplot as plt

N = 50 #size of data
D = 50 #dimension

X = (np.random.random((N,D)) - 0.5) * 10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3)) #true weights

Y = X.dot(true_w) + np.random.randn(N)*0.5 #model add some white noise

costs = []
w = np.random.randn(D)/np.sqrt(D) #initialize random weights
learning_rate = 0.001
l1 = 10.0 #l1 regulariozation size
for t in xrange(500):
    Yhat = X.dot(w)#estimated Y
    delta = Yhat - Y #discrepancy
    #gradient descent step
    w = w - learning_rate*(X.T.dot(delta)+ l1*np.sign(w))
    mse = delta.dot(delta) / N #mse error
    costs.append(mse)# record mse through iterations

plt.plot(costs)
plt.show()

print "final w:", w
plt.plot(true_w, label = 'true w')
plt.plot(w, label = "w_map")
plt.legend()

#L1 + L2 = elastic net
