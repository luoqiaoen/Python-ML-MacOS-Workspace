import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)#2 rows of random array
# ones = np.array([[1]*N]).T #1 row of ones
# Xb = np.concatenate((ones,X), axis=1)#combined
# w = np.random.randn(D+1)#three random ws
# z = Xb.dot(w)#output
# def sigmoid(z):
#     return 1/(1+np.exp(-z))
#
# print sigmoid(z)

X[:50,:] = X[:50,:] - 2*np.ones((50,D)) #first 50 points
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) #the rest of N-50 points
ones = np.array([[1]*N]).T #1 row of ones

T = np.array([0]*50+[1]*50) #target, first 50 are zeros, the rest 1s

w = np.random.randn(D+1)#three random weights
Xb = np.concatenate((ones,X), axis=1)#combined

z = Xb.dot(w)


def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T,Y):
    E = 0
    for i in xrange(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

print cross_entropy(T,Y)

w = np.array([0,4,4]) #closed-form solution

z = Xb.dot(w)
Y = sigmoid(z)

print cross_entropy(T,Y)
