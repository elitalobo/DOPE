from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sklearn
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# In[7]:







class PolynomialBasis():
    def __init__(self, deg=1):
        print(deg)

        self.poly = PolynomialFeatures(deg)

    def transform(self, x):
        if len(x.shape)==1:
            x = x.reshape(1,-1)
        x = self.poly.fit_transform(x)
        return x[:,1:]

import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time
import torch
import numpy as np


def get_distance_tensor(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return torch.sqrt(sum)

def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def kmeans(X, k, max_iters):
    X = X.numpy()
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    converged = False

    current_iter = 0

    while (not converged) and (current_iter < max_iters):

        cluster_list = [[] for i in range(len(centroids))]

        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            distances_list = np.array(distances_list)
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))

        prev_centroids = centroids.copy()

        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        print('K-MEANS: ', int(pattern))

        converged = (pattern == 0)

        current_iter += 1

    return np.array(centroids), np.array([np.std(x) for x in cluster_list])
class RBFNetwork:

    def __init__(self, X, y, tX, ty, num_of_classes,
                 k, std_from_clusters=True):
        self.X = X
        self.y = y

        self.tX = tX
        self.ty = ty

        self.number_of_classes = num_of_classes
        self.k = k
        self.std_from_clusters = std_from_clusters

    def convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return 1 / np.exp(-distance / s ** 2)

    def rbf_list(self, X, centroids, std_list):
        RBF_list = []
        for x in X:
            RBF_list.append([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
        return np.array(RBF_list)

    def fit(self):

        self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=1000)

        if not self.std_from_clusters:
            dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)

        self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)

        RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)

        self.pred_ty = RBF_list_tst @ self.w

        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

        diff = self.pred_ty - self.ty

        print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))


def _get_order_array(order,number_of_states,start = 0):
    arr = []
    for i in itertools.product(np.arange(start,order + 1),repeat=(number_of_states)):
        arr.append(np.array(i))
    return np.array(arr)

def fourier_basis(state, order_list):
    '''
    Convert state to order-th Fourier basis
    '''

    state_new = np.array(state).reshape(1,-1)
    scalars = np.einsum('ij, kj->ik', order_list, state_new)
    phi = np.cos(np.pi*scalars)
    return phi



def radial_basis_function(state,order_list,order,sigma):

    state = np.array(state).reshape(1, -1)
    #c = order_list * (1/order)
    c = order_list
    subs = np.subtract(c,state)
    #sigma = 2/(order-1)
    norms_squared = np.power(np.linalg.norm(subs,axis=1,keepdims=True),2)

    a_k = np.exp(-norms_squared / (sigma * 2))*(1/np.sqrt(2*np.pi*sigma))
    phi = a_k
    #phi = a_k / np.sum(a_k)
    assert phi.shape == (len(order_list), 1)
    return phi



class RBFBasis():

    def __init__(self,
                 k, std_from_clusters=True):

        self.k = k
        self.X = None
        self.std_from_clusters = std_from_clusters


    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return 1 / np.exp(-distance / s ** 2)

    def rbf_list(self, X, centroids, std_list):
        RBF_list = []
        for x in X:
            RBF_list.append(np.array([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)]))
        return np.array(RBF_list)

    def transform(self, X):
        if X.shape[0]==1 or self.X is not None:
            RBF_X = self.rbf_list(X, self.centroids, self.std_list)
            return RBF_X


        else:

            self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=2000)

            if not self.std_from_clusters:
                dMax = np.max(np.array([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids]))
                self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)


            RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)
            return RBF_X



