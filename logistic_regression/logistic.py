import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)#2 rows of random array
ones = np.array([[1]*N]).T #1 row of ones
Xb = np.concatenate((ones,X), axis=1)#combined
w = np.random.randn(D+1)#three random ws
z = Xb.dot(w)#output
def sigmoid(z):
    return 1/(1+np.exp(-z))

print sigmoid(z)
