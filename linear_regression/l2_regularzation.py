# note L2 regulariozation assumes prior distribution is Gaussian, L1 is Laplacean

import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0,10,N)

Y = 0.5*X + np.random.randn(N)

#last two data are outliers
Y[-1] += 30
Y[-2] += 30

plt.scatter(X,Y)
plt.show()

#add the bias terms
X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml)
plt.show()

l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2)+X.T.dot(X), X.T.dot(Y))
# note that L2 can be direcly solved but not L1
Yhat_map = X.dot(w_map)

plt.scatter(X[:,1],Y)
plt.scatter(X[:,1],Yhat_ml, label = 'maximum likelihood')
plt.scatter(X[:,1],Yhat_map, label = 'maximum a priori')

plt.legend()
plt.show()
