#2 hidden layers nn using tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import get_normalized_data, y2indicator

def error_rate(p,t):
    return np.mean(p!=t)

def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    max_iter = 20
    print_period =10
    lr = 0.0001
    reg = 0.01
    Xtrain = Xtest.astype(np.float32)
    Ytrain = Ytest.astype(np.float32)
    Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
    Ytest_ind = y2indicator(Ytest).astype(np.float32)
    N,D = Xtrain.shape
    batch_sz = 500
    n_batches = N//batch_sz

    M1 = 300
    M2 = 100
    K = 10
    W1_int = np.random.randn(D,M1)/28
    b1_int = np.zeros(M1)
    W2_int = np.random.randn(M1,M2)/np.sqrt(M1)
    b2_int = np.zeros(M2)
    W3_int = np.random.randn(M2,K)/np.sqrt(M2)
    b3_int = np.zeros(K)
    # TensorFlow variables == Theano shared variables.
    # But Theano variables == TensorFlow placeholders.
    X = tf.placeholder(tf.float32, shape=(None, D), name = 'X')
    T = tf.placeholder(tf.float32, shape= (None, K), name = 'T')
    W1 = tf.Variable(W1_int.astype(np.float32))
    b1 = tf.Variable(b1_int.astype(np.float32))
    W2 = tf.Variable(W2_int.astype(np.float32))
    b2 = tf.Variable(b2_int.astype(np.float32))
    W3 = tf.Variable(W3_int.astype(np.float32))
    b3 = tf.Variable(b3_int.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X,W1)+b1)
    Z2 = tf.nn.relu(tf.matmul(Z1,W2)+b2)
    Yish = tf.matmul(Z2,W3)+b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))
    train_op = tf.train.RMSPropOptimizer(lr, decay = 0.99, momentum = 0.9).minimize(cost)
    predict_op = tf.argmax(Yish, 1)
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)
    LL = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz),]

            sess.run(train_op,feed_dict={X:Xbatch, T:Ybatch})
            if j % print_period == 0:
                test_cost = sess.run(cost, feed_dict={X:Xtest, T:Ytest_ind})
                prediction = sess.run(predict_op,feed_dict={X:Xtest})
                err = error_rate(prediction, Ytest)
                print "cost/err at iteration i = %d, j = %d: %.3f / %.3f \n" % (i, j, test_cost, err),
                LL.append(test_cost)
    plt.plot(LL)
    plt.show()

if __name__=='__main__':
    main()
