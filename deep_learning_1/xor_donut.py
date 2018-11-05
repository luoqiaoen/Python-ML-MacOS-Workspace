import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    # Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))
    # activation = Z.dot(W2) + b2
    # Y = 1/(1+np.exp(-activation))
    # return Y,Z
    #relu
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)
    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z

def predict(X,W1,b1,W2,b2):
    Y,_ = forward(X,W1,b1,W2,b2)
    return np.round(Y)

def derivative_w2(Z, T, Y):
    return (T-Y).dot(Z)

def derivative_b2(T, Y):
    return (T-Y).sum()

def derivative_w1(X,Z,T,Y,W2):
    #sigmmoid derivative
    # dZ = np.outer(T-Y,W2)*Z*(1-Z)
    #tanh derivative
    #dZ = np.outer(T-Y, W2)*(1-Z*Z)
    #relu
    dZ = np.outer(T-Y, W2) * (Z > 0)
    return X.T.dot(dZ)

def derivative_b1(Z,T,Y,W2):
    ##tanh
    #dZ = np.outer(T-Y,W2)*(1-Z*Z)
    #relu
    dZ = np.outer(T-Y, W2) * (Z > 0)
    return dZ.sum(axis=0)

def cost(T,Y):
    # tot = 0
    # #cross entropy
    # for n in xrange(len(T)):
    #     if T[n] == 1:
    #         tot+= np.log(Y[n])
    #     else:
    #         tot += np.log(1 - Y[n])
    # return tot
    # log likilhood
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 5)
    b1 = np.zeros(5)
    W2 = np.random.randn(5)
    b2 = np.random.randn(1)
    LL = [] #liklehood tracking
    learning_rate = 0.0005
    regularization = 0.
    last_error_rate = None
    for i in xrange(30000):
        pY,Z = forward(X,W1,b1,W2,b2)
        ll = cost(Y,pY)
        prediction = predict(X,W1,b1,W2,b2)
        er = np.mean(prediction != Y)
        if LL and ll < LL[-1]:
            print "early exit"
            break
        LL.append(ll)
        W2 += learning_rate*(derivative_w2(Z,Y,pY)-regularization*W2)
        b2 += learning_rate*(derivative_b2(Y,pY)-regularization*b2)
        W1 += learning_rate*(derivative_w1(X,Z,Y,pY,W2)-regularization*W1)
        b1 += learning_rate*(derivative_b1(Z,Y,pY,W2) - regularization*b1)
        if i % 300 == 0:
            print ll
    print "final classification rate: ", 1 - np.abs(prediction-Y).mean()
    plt.plot(LL)
    plt.show()

def test_donut():
    N = 1000
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.random.random(N/2)
    X_inner = np.concatenate([[R1*np.cos(theta)],[R1*np.sin(theta)]]).T

    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.random.random(N/2)
    X_outer = np.concatenate([[R2*np.cos(theta)],[R2*np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*(N/2)+[1]*(N/2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = []
    learning_rate = 0.00005
    regularization = 0.2
    last_error_rate = None
    for i in range(3000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 300 == 0:
            print("i:", i, "ll:", ll, "classification rate:", 1 - er)
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    test_xor()
    #test_donut()
