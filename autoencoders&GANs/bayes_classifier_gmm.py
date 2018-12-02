# bayes classifier with built-in Gaussian mixture
import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class BayesClassifier:
    def fit(self, X, Y):
        #classes are numbered 0 to K-1
        self.K = len(set(Y))
        self.gaussians = []
        for k in range(self.K):
            print("Fitting gmm", k)
            Xk = X[Y == k]
            gmm = BayesianGaussianMixture(10)
            gmm.fit(Xk)
            self.gaussians.append(gmm)

    def sample_given_y(self,y):
        gmm = self.gaussians[y]
        sample = gmm.sample()
        # note: sample returns a tuple containing 2 things:
        # 1) the sample
        # 2) which cluster it came from
        # we'll use (2) to obtain the means so we can plot
        # them like we did in the previous script
        # we cheat by looking at "non-public" params in
        # the sklearn source code
        mean = gmm.means_[sample[1]]
        return sample[0].reshape(28,28), mean.reshape(28,28)
    def sample(self):
        y = np.random.randint(self.K) #sample y
        return self.sample_given_y(y) #sample x given y

if __name__ == '__main__':
  X, Y = util.get_mnist()
  clf = BayesClassifier()
  clf.fit(X, Y)

  for k in range(clf.K):
      sample, mean = clf.sample_given_y(k)
      plt.subplot(1,2,1)
      plt.imshow(sample, cmap='gray')
      plt.title("Sample")
      plt.subplot(1,2,2)
      plt.imshow(mean, cmap='gray')
      plt.title("Mean")
      plt.show()

  # generate a random sample
  sample, mean = clf.sample()
  plt.subplot(1,2,1)
  plt.imshow(sample, cmap='gray')
  plt.title("Random Sample from Random Class")
  plt.subplot(1,2,2)
  plt.imshow(mean, cmap='gray')
  plt.title("Corresponding Cluster Mean")
  plt.show()
