import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

R1 = np.random.randn(N/2) + R_inner
theta = 2*np.pi*np.random.random(N/2)
X_inner = np.concatenate([[R1*np.cos(theta)],[R1*np.sin(theta)]]).T

R2 = np.random.randn(N/2) + R_outer
theta = 2*np.pi*np.random.random(N/2)
X_outer = np.concatenate([[R2*np.cos(theta)],[R2*np.sin(theta)]]).T

X = np.concatenate([X_inner,X_outer])
T = np.array([0]*(N/2)+[1]*(N/2))

plt.scatter(X[:,0],X[:,1],c=T)
plt.show()

ones = np.array([[1]*N]).T

r = np.zeros((N,1))
for i in xrange(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones,r,X), axis = 1)
w = np.random.rand(D+2)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

learning_rate = 0.0001
error = []
for i in xrange(5000):
    e = cross_entropy(T,Y)
    error.append(e)
    if i % 100 == 0:
        print e
    w += learning_rate*(np.dot((T-Y).T, Xb) - 0.01*w)
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy")

print "Final w:" ,w
print "Final classfication rate:", 1- np.abs(T-np.round(Y)).sum()/N
