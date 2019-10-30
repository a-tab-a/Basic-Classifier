#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math


def gaussian_pdf(mean,std,x):
    a=math.exp(-(x-mean)**2/(2*std**2))
    prob=a/(math.sqrt(2*math.pi*std**2))
    return prob
    
    
class BayesClassifier():
    def __init__(self,data_dict):
        
        """Input data is dictionary, specifying the test_data and train data"""
        self.training_data=data_dict['x']
        self.label=data_dict['y']
        self.test_data=data_dict['x_test']
        self.test_label=data_dict['y_test']
        
        
    def Naive_Bayes(self,K, threshold):
        """K is the number of class in data"""

        x_train=np.array(self.training_data, dtype="float64") > threshold
        x_test=np.array(self.test_data, dtype="float64") > threshold
        n_point, dim = x_train.shape
        ycounts = np.bincount(self.label.flatten())
        y_prior=ycounts/n_point
        theta=np.zeros((K, dim))

        #theta claculated from dirichlet eqn
        for cls in range(K):
            idx=np.where(self.label.flatten() == cls)
            pixel_on = np.sum(x_train[idx], axis=0, dtype="float64") + 1.  #suuming all ;(n+1) dirichlet eqn follow
            theta[cls] += pixel_on / (ycounts[cls] + K) #N+k


        log_theta = np.log(theta)
        log_inver = np.log(1. - theta)
        log_prior = np.log(y_prior)


        y_estimated = np.zeros(self.test_label.shape)

        for i in range(x_test.shape[0]):

            #log likelihood eqn is sum (xnlnu+(1-xn)ln(1-u))
            log_likelihood = np.sum(log_theta[:, x_test[i]], axis=1) + np.sum(log_inver[:, np.logical_not(x_test[i])], axis=1)
            log_posterior = log_prior + log_likelihood
            #i find log posterior for each class and then find where is the max value 
            y_estimated[i] = np.argmax(log_posterior)

        dirichlet_accuray=float(sum(self.test_label == y_estimated)) / self.test_label.shape[0]

        print("Data Classification accuracy using Naive Bayes with Dirichlet prior is",dirichlet_accuray)
        
    def get_specific_class():
        return specific_data_dict


    def Naive_Bayes_Gaussian():
        log_likelihood = np.zeros((y2_test.shape))
        gaussians_class5=np.ones((dim))
        gaussians_all=np.ones((dim))


        for i in range(x2_test.shape[0]):

            for j in range(dim):
                gaussians_class5[j] = gaussian_pdf(mean[0, j], standard_dev0,x2_test[i,j])
                gaussians_all[j] = gaussian_pdf(mean[1, j], standard_dev1,x2_test[i,j])
                
            log_likelihood_5 = np.sum(np.log(gaussians_class5))
            log_likelihood_all = np.sum(np.log(gaussians_all))
            log_likelihood[i] = log_likelihood_5 - log_likelihood_all

        x_data_plot = []
        y_data_plot = []
        y_hat = np.zeros((y2_test.shape))


        for tau in range(100):
            mask = (log_likelihood > tau)
            y_hat[mask] = 1
            y_hat[np.logical_not(mask)] = 0
            true_pos_mask = np.where((np.logical_and(y2_test.flatten() == 1, y_hat.flatten() == 1)))
            true_neg_mask = np.where(np.logical_and(y2_test.flatten() == 0, y_hat.flatten() == 0))
            false_pos_mask = np.where(np.logical_and(y2_test.flatten() == 0, y_hat.flatten() == 1))
            false_neg_mask = np.where(np.logical_and(y2_test.flatten() == 1, y_hat.flatten() == 0))
            FPR=float(len(false_pos_mask[1])/(len(true_neg_mask[1])+len(false_pos_mask[1])))
            TPR=float(len(true_pos_mask[1])/(len(true_pos_mask[1])+len(false_neg_mask[1])))
            x_data_plot.append(FPR)
            y_data_plot.append(TPR)
        
        
        plt.plot(x_data_plot, y_data_plot)
        plt.title("ROC Diagram")
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.show()