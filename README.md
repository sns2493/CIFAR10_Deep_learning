knn_model: Implements training of 50000 Images from CIFAR-10 dataset and testing on 10000 images using the K-Nearest-Neighbour algorithm for different values of k.
1. Required packages: cPickle, numpy, os, sklearn and time
2. Change the address of the dataset in the variable "data_path" in the code.

svm_model: Implements training of 50000 Images from CIFAR-10 dataset and testing on 10000 images using different kernels of SVM algorithm.
1. Required packages: cPickle, numpy, os, sklearn and time
2. Change the address of the dataset in the variable "data_path" in the code.

cnn_model_final: Implements training of 50000 Images from CIFAR-10 dataset and testing on 10000 images using a Convoluted Neural Network
1. Required packages: keras, numpy(latest version)
2. Use the "tensorflow" backend in keras.
3. ZCA whitening is turned off by default since it takes lot of time(>5 hours) with it turned on on the HPC. You can turn it on by changing the value of zca_whitening flagto TRUE in line 25.