#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def extract_MNISTdata(filename):

    training_image_file=open(filename,'rb')
    training_images=training_image_file.read()
    training_image_file.close()
    training_images=bytearray(training_images);training_images=training_images[16:]
    training_images = np.frombuffer(training_images, dtype=np.uint8).astype(np.float32)
    data=training_images.shape[0]
    image_size = 28*28
    num_of_images =int(data/image_size)
    training_images=training_images.reshape(num_of_images, image_size)

    #plt.imshow(training_images[2].reshape(28, 28))
    #plt.show()
    return training_images


def MNIST_labels(filename):
  
    training_labels_file=open(filename,'rb')
    training_labels = training_labels_file.read()
    training_labels_file.close()
    training_labels = bytearray(training_labels);training_labels = training_labels[8:]
    training_labels = np.array(training_labels);num_of_labels = training_labels.shape[0]
    return training_labels.reshape(num_of_labels, 1) 




x = extract_MNISTdata('train-images.idx3-ubyte')
y = MNIST_labels('train-labels.idx1-ubyte')
x_test = extract_MNISTdata('t10k-images.idx3-ubyte')
y_test = MNIST_labels('t10k-labels.idx1-ubyte')

mnist_dict={
                'x':x,
                'y': y,
                'x_test': x_test,
                'y_test': y_test}
                
from Bayes_Classifier import BayesClassifier



k=BayesClassifier(mnist_dict)
classify=k.Naive_Bayes(10,100)
pl=k.Naive_Bayes_Gaussian(5)