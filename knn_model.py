import cPickle as pickle
import numpy as np
import os
from sklearn import svm
from sklearn import neighbors, datasets
import time

start_time = time.time()
np.set_printoptions(threshold=np.inf)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data_path = "/home/sshah11/.keras/datasets/cifar-10-batches-py" #insert your path of cifar-10 dataset directory here
X = np.empty([0,3072])
Y = []
for i in range(1,6):
	batch_path = data_path + "/data_batch_" + str(i)
	XTrain = unpickle(batch_path)
	X1 = XTrain['data'].astype("float")
	Y1 = XTrain['labels']
	X = np.concatenate((X,X1), axis = 0)
	Y = Y+Y1

print(X.shape)
print(len(Y))

test_path = data_path + "/test_batch"
XTest = unpickle(test_path)
TX = XTest['data'].astype("float")
TY = XTest['labels']


#centralize data
mean_image = np.mean(X, axis = 0)
X -= mean_image

mean_test = np.mean(TX, axis = 0)
TX -= mean_test
print("dataset preparation--- %s seconds ---" % (time.time() - start_time))
for n_neighbors in range(1,11):
	start_time = time.time()
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance").fit(X, Y)
	print("Training--- %s seconds ---" % (time.time() - start_time))
	start_time = time.time()

#	out = clf.predict(TX)
#	print("predicted values for k = %d:" %(n_neighbors))
#	print(out)

	scr = clf.score(TX, TY)
	print('for k = %d, the accuracy is: %.4f\n' % (n_neighbors, scr))

	print("Testing--- %s seconds ---\n" % (time.time() - start_time))