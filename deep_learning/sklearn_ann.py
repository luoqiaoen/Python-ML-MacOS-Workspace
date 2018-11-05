#Simple sklearn example
from process import get_data
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

X, Y = get_data()

X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

#set model parameters, see MLPClassifier help doc for all features
model = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=2000)

model.fit(Xtrain, Ytrain)

train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print "train accuracy: ", train_accuracy, "test accuracy: ", test_accuracy
