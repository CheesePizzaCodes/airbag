from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

import preprocess
from graphics import plot_histograms



def fit_gaussian(data):
    """
    Computes the mean variance and covariance matrix
    """
    mean = data.mean(axis=0)
    covm = np.cov(data, rowvar=False)
    return mean, covm


def remove_colinear_features(data):
    corrcoef = np.corrcoef(data, rowvar=False)
    corrcoef[np.diag_indices_from(corrcoef)] = 0
    colinear_features = np.where(np.logical_and(corrcoef >= 0.8, corrcoef < 1.))

    colinear_feature_pairs_unique = unique_feature_pairs(colinear_features)

    for i_0, i_1 in colinear_feature_pairs_unique:
         
        resulting_feature = (data[:, i_0] + data[:, i_1]) / 2
        
        data[:, i_0] = resulting_feature  # replace first one by the calculated average

        data = np.delete(data, i_1, axis=1)  # delete the second one

    return data
    ...

def unique_feature_pairs(colinear_features):
    temp = [set(i) for i in zip(*colinear_features)]
    temp = set([tuple(i) for i in temp])
    return temp


def compute_likelihood(datapoint, mu, sigma):
    distribution = multivariate_normal(mu, sigma, allow_singular=False)
    return distribution.pdf(datapoint)


if __name__ == '__main__':
    # import data
    data = np.load('./py_src/data/out.npy')
    data =  np.nan_to_num(data, nan=0)  

    # data = preprocess.batch_preprocess()
    X = data[:, :-1]
    y = data[:, -1:].flatten()

    # X_train, X_test, y_train, y_test = train_test_split(X, y ,shuffle=False)

    X = remove_colinear_features(X)


    negative_cases = X[y == 0]  # normal activities
    positive_cases = X[y == 1]  # falls

    mu, sigma = fit_gaussian(negative_cases)



    # fig, ax = plot_histograms(negative_cases)
    fig2, ax2 = plot_histograms(data_list=[positive_cases, negative_cases])

    likelihoods_0 = compute_likelihood(negative_cases, mu, sigma)
    likelihoods_1 = compute_likelihood(positive_cases, mu, sigma)

    input()
    ...