import numpy as np
import matplotlib.pyplot as plt

#load data
X = []
Y = []

for line in open('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

#np arrays
X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

#calculate a and b in linear regression
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denominator
b = (Y.mean()*X.dot(X) - X.mean()*Y.dot(X))/denominator

Yhat = a*X + b
plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)

print "the  r-squred is: ", r2
