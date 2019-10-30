#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
import random 


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
        
    def get_specific_class(self,class_label):
        
        mask_temp=np.where(self.label.flatten()==class_label);mask_rand=np.random.choice(mask_temp[0],size=1000,replace=False);
        x_cls=self.training_data[mask_rand,:]
        mask_other=np.where(self.label.flatten()!=class_label); mask_other_rand=np.random.choice(mask_other[0],size=1000,replace=False)
        x_all=self.training_data[mask_other_rand,:]
        x_temp=np.vstack((x_cls,x_all));y_temp=np.matrix(np.hstack((np.ones((1000)),np.zeros((1000))))).T;
        dataset=np.hstack((x_temp,y_temp));np.random.shuffle(dataset)
        x=dataset[:1800,:784];x_test=dataset[1800:,:784]
        y=dataset[:1800,784];y_test=dataset[1800:,784]

        specific_data_dict={
                'x':x,
                'y': y,
                'x_test': x_test,
                'y_test': y_test}
                
                
        x_mask=x[np.where(y==1)[0],:]
        x_other=x[np.where(y==0)[0],:]
        y_mask=np.where(y==1)[0]
        y_other=np.where(y==0)[0]


        k=2 #since only specific class need to determine
        dim=x.shape[1]
        mean=np.zeros((k,dim))
        stdv=np.zeros((k))
        mean[0]=np.mean(x_mask,axis=0,dtype="float64") #class 5
        mean[1]=np.mean(x_other,axis=0,dtype="float64")
        stdv[0]=np.std(x_mask,dtype="float64")
        stdv[1]=np.std(x_other,dtype="float64")
        
        return specific_data_dict,mean,stdv



    def Naive_Bayes_Gaussian(self,class_label):
    
        data,mean,stdv=self.get_specific_class(class_label)
        x=data['x']
        y=data['y']
        x_test=data['x_test']
        y_test=data['y_test']

        dim=x.shape[0]
        log_likelihood = np.zeros((y_test.shape))
        gaussians_pivot=np.ones((dim))
        gaussians_other=np.ones((dim))
    

        for i in range(x_test.shape[0]):
            
            
            for j in range(dim):
                gaussians_pivot[j] = gaussian_pdf(mean[0, j], stdv[0],x_test[i,j])
                gaussians_other[j] = gaussian_pdf(mean[1, j], stdv[1],x_test[i,j])
                
            log_likelihood_pivot = np.sum(np.log(gaussians_pivot))
            log_likelihood_other = np.sum(np.log(gaussians_other))
            log_likelihood[i] = log_likelihood_pivot - log_likelihood_other

        #Plotting ROC Curve#
        x_data_plot = []
        y_data_plot = []
        y_hat = np.zeros((y_test.shape))


        for tau in range(100):
            mask = (log_likelihood > tau)
            y_hat[mask] = 1
            y_hat[np.logical_not(mask)] = 0
            true_pos_mask = np.where((np.logical_and(y_test.flatten() == 1, y_hat.flatten() == 1)))
            true_neg_mask = np.where(np.logical_and(y_test.flatten() == 0, y_hat.flatten() == 0))
            false_pos_mask = np.where(np.logical_and(y_test.flatten() == 0, y_hat.flatten() == 1))
            false_neg_mask = np.where(np.logical_and(y_test.flatten() == 1, y_hat.flatten() == 0))
            FPR=float(len(false_pos_mask[1])/(len(true_neg_mask[1])+len(false_pos_mask[1])))
            TPR=float(len(true_pos_mask[1])/(len(true_pos_mask[1])+len(false_neg_mask[1])))
            x_data_plot.append(FPR)
            y_data_plot.append(TPR)
        
        
        plt.plot(x_data_plot, y_data_plot)
        plt.title("ROC Diagram")
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.show()