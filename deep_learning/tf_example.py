import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Generate Random Training Data
Nclass = 500
D = 2 #input dimension
M = 3 #layer size
K = 3 #number of classes

X1 = np.random.randn(Nclass, D) + np.array([0,-2])
X2 = np.random.randn(Nclass, D) + np.array([2,2])
X3 = np.random.randn(Nclass, D) + np.array([-2,2])
X = np.vstack([X1,X2,X3]).astype(np.float32)
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

N = len(Y)
T = np.zeros((N,K))
for i in xrange(N):
    T[i,Y[i]] = 1

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev= 0.01))

tfX = tf.placeholder(tf.float32,shape=(None, D),name='tfX')
tfY = tf.placeholder(tf.float32,shape=(None, K),name = 'tfY')
W1 = init_weights([D,M])
b1 = init_weights([M])
W2 = init_weights([M,K])
b2 = init_weights([K])

Z1 = tf.nn.sigmoid( tf.matmul(tfX, W1) + b1 )
py_x = tf.matmul(Z1, W2) + b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=tfY))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x,1)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

for i in xrange(1000):
    sess.run(train_op, feed_dict = {tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    if i % 10 == 0:
        print np.mean(Y == pred)
