import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
X = np.random.randn(N,D)

#data
X[:50,:] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

#labels
T = np.array([0]*50 + [1]*50)

#add a column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X), axis=1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

w = np.array([0,4,4])

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6,6,100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
