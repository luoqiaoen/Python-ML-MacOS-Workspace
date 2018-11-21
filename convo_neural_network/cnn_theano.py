#mat files from http://ufldl.stanford.edu/housenumbers/
#Two CNN layer and 1 hidden layer of vanilla ANN
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

def flatten(X):
    N = X.shape[-1]
    flat = np.zeros((N,3072))
    for i in xrange(N):
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N,10))
    for i in xrange(N):
        ind[i,y[i]] = 1
    return ind

def error_rate(p,t):
    return np.mean(p!=t)

def relu(a):
    return a*(a>0)

def convpool(X,W,b,poolsize=(2,2)):
    conv_out = conv2d(input=X, filters=W)
    pooled_out = pool.pool_2d(
        input=conv_out,
        ws=poolsize,
        ignore_border=True)
    return T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

def init_filter(shape, poolsize):
    w = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:])+shape[0]*np.prod(shape[2:]/np.prod(poolsize)))
    return w.astype(np.float32)

def rearrange(X):
    N = X.shape[-1]
    out = np.zeros((N,3,32,32), dtype=np.float32)
    for i in xrange(N):
        for j in xrange(3):
            out[i,j,:,:] = X[:,:,j,i] # reorder matlab mfile matrix
    return out/255

def main():
    train = loadmat('train_32x32.mat')
    test = loadmat('test_32x32.mat')
    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    Xtest  = rearrange(test['X'])
    Ytest  = test['y'].flatten() - 1
    del test

    Ytest_ind = y2indicator(Ytest)

    max_iter = 6
    print_period = 10

    lr = np.float32(1e-2)
    mu = np.float32(0.99)

    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz

    M = 500
    K = 10
    poolsz = (2, 2)

    # after conv will be of dimension 32 - 5 + 1 = 28
    # after downsample 28 / 2 = 14
    W1_shape = (20, 3, 5, 5) # (num_feature_maps, num_color_channels, filter_width, filter_height)
    W1_init = init_filter(W1_shape, poolsz)
    b1_init = np.zeros(W1_shape[0], dtype=np.float32) # one bias per output feature map

    # after conv will be of dimension 14 - 5 + 1 = 10
    # after downsample 10 / 2 = 5
    W2_shape = (50, 20, 5, 5) # (num_feature_maps, old_num_feature_maps, filter_width, filter_height)
    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[0], dtype=np.float32)

    # vanilla ANN weights
    W3_init = np.random.randn(W2_shape[0]*5*5, M) / np.sqrt(W2_shape[0]*5*5 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)

    # define theano variables and expressions
    X = T.tensor4('X', dtype='float32')
    Y = T.ivector('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
    W3 = theano.shared(W3_init.astype(np.float32), 'W3')
    b3 = theano.shared(b3_init, 'b3')
    W4 = theano.shared(W4_init.astype(np.float32), 'W4')
    b4 = theano.shared(b4_init, 'b4')

    # forward pass
    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)
    pY = T.nnet.softmax( Z3.dot(W4) + b4)

    # define the cost function and prediction
    cost = -(T.log(pY[T.arange(Y.shape[0]), Y])).mean()
    prediction = T.argmax(pY, axis=1)

    # training expressions and functions
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    # momentum changes
    dparams = [
        theano.shared(
            np.zeros_like(
                p.get_value(),
                dtype=np.float32
            )
        ) for p in params
    ]

    updates = []
    grads = T.grad(cost, params)
    for p, dp, g in zip(params, dparams, grads):
        dp_update = mu*dp - lr*g
        p_update = p + dp_update

        updates.append((dp, dp_update))
        updates.append((p, p_update))

    train = theano.function(
        inputs=[X, Y],
        updates=updates,
    )

    # create another function for this because we want it over the whole dataset
    get_prediction = theano.function(
        inputs=[X, Y],
        outputs=[cost, prediction],
    )

    t0 = datetime.now()
    costs = []
    for i in range(max_iter):
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),]

            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest)
                err = error_rate(prediction_val, Ytest)
                print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
                costs.append(cost_val)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()
