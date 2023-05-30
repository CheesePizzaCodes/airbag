from typing import List
import sys

import numpy as np
from numpy.linalg import inv, det

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from scipy.special import log_expit, expit

import preprocess
from graphics import plot_histograms_grid, plot_superimposed_histograms, plot_pairwise_scatter


def fit_gaussian(data):
    """
    Computes the mean variance and covariance matrix
    """
    mean = data.mean(axis=0)
    covm = np.cov(data, rowvar=False)
    return mean, covm


def reduce_data_dimensionality(data: np.ndarray, remaining_dims: int = None, discarded_dims: int = None):
    if remaining_dims is not None:
        dimensionality = remaining_dims
    elif discarded_dims is not None:
        dimensionality = data.shape[1] - discarded_dims
    else:
        raise ValueError
    pca = PCA(dimensionality)
    return pca.fit_transform(data)


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


def compute_likelihood(datapoint: np.ndarray, mu: np.ndarray,
                       sigma: np.ndarray) -> np.ndarray:  # TODO implement some other measure so that it's easier to set epsilon
    distribution = multivariate_normal(mu, sigma, allow_singular=False)
    likelihood = distribution.pdf(datapoint)

    likelihood = np.clip(likelihood, a_min=sys.float_info.epsilon, a_max=None)


    return np.log(likelihood)




def classify_datapoints(likelihoods: np.ndarray, eps: float):
    return likelihoods < eps


def compute_accuracy(predicted_values: np.ndarray, ground_truth: np.ndarray):
    return (predicted_values == ground_truth).mean()


def main(eps):
    # import data
    data = np.load('./py_src/data/ws25_ol0.5--.npy')
    data = np.nan_to_num(data, nan=0)

    # data = preprocess.batch_preprocess()
    X = data[:, :-1]
    y = data[:, -1:].flatten()

    # X_train, X_test, y_train, y_test = train_test_split(X, y ,shuffle=False)



    # X = reduce_data_dimensionality(X, remaining_dims=10)

    negative_cases = X[y == 0.]  # normal activities
    positive_cases = X[y == 1.]  # falls

    mu, sigma = fit_gaussian(negative_cases)

    # fig2, ax2 = plot_histograms_grid(data_list=[negative_cases, positive_cases], num_rows=X.shape[1] // 9)

    likelihoods_0 = compute_likelihood(negative_cases, mu, sigma)
    likelihoods_1 = compute_likelihood(positive_cases, mu, sigma)


    # plot_superimposed_histograms([likelihoods_0, likelihoods_1])

    # eps = 10
    likelihoods = compute_likelihood(X, mu, sigma)
    predicted_classes = classify_datapoints(likelihoods, eps)

    # acc = compute_accuracy(predicted_classes, y)

    f1 = f1_score(y, predicted_classes)
    prec = precision_score(y, predicted_classes)
    rec = recall_score(y, predicted_classes)

    print(f'{eps}, {f1:.4}, {prec:.4}, {rec:.4}')

    pts = (eps, f1)

    return pts


if __name__ == '__main__':
    main(20)
    pts = []
    print('eps, f1, prec, rec')
    for i in np.arange(10, 210, 10):
        pts.append(main(i))

    plt.plot(*list(zip(*pts)))
    plt.show()
