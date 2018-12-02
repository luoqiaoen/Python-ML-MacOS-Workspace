# one hidden layer Autoencoder M to D to M
import util
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, D, M):
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        self.W = tf.Variable(tf.random_normal(shape=(D,M))*2/np.sqrt(M))
        self.b = tf.Variable(np.zeros(M).astype(np.float32))

        #input to hidden
        self.V = tf.Variable(tf.random_normal(shape=(M,D))*2/np.sqrt(D))
        self.c = tf.Variable(np.zeros(D).astype(np.float32))

        #the reconstruction
        self.Z = tf.nn.relu(tf.matmul(self.X, self.W)+self.b)
        logits = tf.matmul(self.Z, self.V) + self.c
        self.X_hat = tf.nn.sigmoid(logits)

        self.cost = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = self.X,
                logits = logits
            )
        )
        #train
        self.train_op = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(self.cost)
        #session setup
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def fit(self, X, epochs=30, batch_sz=64):
        costs = []
        n_batches = len(X)//batch_sz
        print("n batches:", n_batches)
        for i in range(epochs):
            print("epoch:", i)
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j+1)*batch_sz]
                _,c, = self.sess.run((self.train_op,self.cost),feed_dict = {self.X: batch})
                c /= batch_sz
                costs.append(c)
                if j%100 ==0:
                    print("iter: %d, cost: %.3f" % (j, c))
        plt.plot(costs)
        plt.show()
    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X: X})

def main():
    X, Y = util.get_mnist()

    model = Autoencoder(784, 300)
    model.fit(X)

    for i in range(len(X)):
        if i % 10 ==0:
            x = X[i]
            im = model.predict([x]).reshape(28,28)
            plt.subplot(1,2,1)
            plt.imshow(x.reshape(28,28), cmap='gray')
            plt.title("Original")
            plt.subplot(1,2,2)
            plt.imshow(im,cmap='gray')
            plt.title("Reconstruction")
            plt.show()

if __name__ == '__main__':
    main()
