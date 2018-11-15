import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle
from util import getImageData, error_rate, init_weight_and_bias, y2indicator

class HiddenLayer(object):
	def __init__(self,M1,M2,an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(M1,M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self,X):
		return tf.nn.relu(tf.matmul(X,self.W)+self.b)

def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return w.astype(np.float32)

class ConvPoolLayer(object):
    def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
        sz = (fw, fh, mi, mo)
        W0 = init_filter(sz,poolsz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.poolsz = poolsz
        self.params = [self.W, self.b]

    def forward(self, X):
        conv_out = tf.nn.conv2d(X,self.W,strides=[1,1,1,1], padding = 'SAME')
        conv_out = tf.nn.bias_add(conv_out,self.b)
        pool_out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        return tf.tanh(pool_out)


class CNN(object):
    def __init__(self,convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, lr=10e-4, mu=0.99, reg =10e-4, decay = 0.999999, eps=10e-3, batch_sz=30, epochs=3, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        K = len(set(Y))

        X, Y = shuffle(X,Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)

        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y =X[:-1000],Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1)
        N, d, d, c = X.shape
        mi = c
        outw = d
        outh = d
        self.convpol_layers = []
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi,mo,fw,fh)
            self.convpol_layers.append(layer)
            outw = outw/2
            outh = outh/2
            mi = mo

        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1,M2,count)
            self.hidden_layers.append(h)
            M1 = M2
            count+=1

        W,b = init_weight_and_bias(M1,K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')

        self.params = [self.W, self.b]
        for h in self.convpol_layers:
            self.params += h.params
        for h in self.hidden_layers:
            self.params += h.params

        tfX = tf.placeholder(tf.float32, shape=(None,d,d,c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None,K), name='Y')
        act = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
        	tf.nn.softmax_cross_entropy_with_logits_v2(logits=act,labels=tfY)+rcost
        )
        prediction = self.predict(tfX)
        train_op = tf.train.RMSPropOptimizer(lr, decay = decay, momentum = mu).minimize(cost)
        n_batches = N//batch_sz
        costs = []
        init = tf.initialize_all_variables()

        with tf.Session() as session:
            session.run(init)
            for i in xrange(epochs):
                X, Y = shuffle(X,Y)
                for j in xrange(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]
                    session.run(train_op, feed_dict = {tfX:Xbatch, tfY:Ybatch})
                    if j % 20 == 0:
                        c = session.run(cost, feed_dict = {tfX:Xvalid, tfY:Yvalid})

                        p = session.run(prediction, feed_dict = {tfX:Xvalid, tfY:Yvalid})
                        err = error_rate(Yvalid_flat, p)
		        print('cost/err at iteration i=%d j=%d nb=%d is %.3f/%.3f'%(i,j,n_batches,c,err))
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for c in self.convpol_layers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z,self.W)+self.b

    def predict(self,X):
        pY = self.forward(X)
        return tf.argmax(pY,1)

def main():
    X, Y = getImageData()
    X = X.transpose((0,2,3,1))
    model = CNN(convpool_layer_sizes=[(20,5,5),(20,5,5)],
                hidden_layer_sizes=[500,300],)
    model.fit(X,Y,show_fig=True)

if __name__ == '__main__':
    main()
