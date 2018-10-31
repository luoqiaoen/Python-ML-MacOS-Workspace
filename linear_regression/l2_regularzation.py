# note L2 regulariozation assumes prior distribution is Gaussian, L1 is Laplacean

import numpy as np
import matplotlib.pyplot as plt

N = 50 #size of data

X = np.linspace(0,10,N)

Y = 0.5*X + np.random.randn(N) #add noise

#last two data are outliers
Y[-1] += 30
Y[-2] += 30

plt.scatter(X,Y)
plt.show()

#add the bias terms
X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))#maximum liklihood
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml)
plt.show()

l2 = 1000.0
# MAP picture, solve (\lambda I + X^T*X) w = X^T*Y
# note that L2 can be direcly solved but not L1
w_map = np.linalg.solve(l2*np.eye(2)+X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)

plt.scatter(X[:,1],Y)
plt.scatter(X[:,1],Yhat_ml, label = 'maximum likelihood')
plt.scatter(X[:,1],Yhat_map, label = 'maximum a priori')

plt.legend()
plt.show()
