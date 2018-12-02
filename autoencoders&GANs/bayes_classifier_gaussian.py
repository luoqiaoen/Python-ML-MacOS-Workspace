import util
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

class BayesClassifier:
    def fit(self, X, Y):
        #classes are numbered 0 to K-1
        self.K = len(set(Y))
        self.gaussians = []

        for k in range(self.K):
            Xk = X[Y==k]
            mean = Xk.mean(axis = 0)
            cov = np.cov(Xk.T)
            g = {'m':mean, 'c': cov}
            self.gaussians.append(g) #get the distributions
    def sample_given_y(self,y):
        g = self.gaussians[y]
        return mvn.rvs(mean=g['m'], cov = g['c'])
    def sample(self):
        y = np.random.randint(self.K) #sample y
        return self.sample_given_y(y) #sample x given y

# show one sample for each class
# also show the mean image learned
if __name__ == '__main__':
    X, Y = util.get_mnist()
    clf = BayesClassifier()
    clf.fit(X,Y)

    for k in range(clf.K):
        sample = clf.sample_given_y(k).reshape(28,28)
        mean = clf.gaussians[k]['m'].reshape(28,28)

        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean,cmap='gray')
        plt.title("Mean")
        plt.show()

    sample = clf.sample().reshape(28,28)
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample from Random Class")
