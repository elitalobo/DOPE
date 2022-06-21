import sys
sys.path.append("experiments/")
from argparser_fqe import parse

args = parse()
import torch

torch.manual_seed(args.seed)
import random

random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# In[7]:







class PolynomialBasis():
    def __init__(self, deg=1, interactive=False):
        print(deg)

        self.poly = PolynomialFeatures(deg, interaction_only=interactive)

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
    X = (x1-x2.reshape(1,-1))**2
    X = torch.sum(X,dim=1)
    return torch.sqrt(X+1e-4)

def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def kmeans(X, k, max_iters):
    #X = X.cpu().numpy()
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



class RBFBasis:

    def __init__(self,
                 k, std_from_clusters=True):

        self.k = k
        self.X = None
        self.std_from_clusters = std_from_clusters


    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return np.exp(-distance / s ** 2)

    def inverse_rbf(self,x,c,s):
        pass

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
            self.X = X

            self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=2000)

            if not self.std_from_clusters:
                dMax = np.max(np.array([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids]))
                self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)


            RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)
            return RBF_X

    def inverse_transform(self,X):
        RBF_list = []
        for x in X:
            RBF_list.append(np.array([self.inverse_rbf(x, c, s) for (c, s) in zip(self.centroids, self.std_list)]))
        return np.array(RBF_list)


class PolynomialBasisTensor():
    def __init__(self, deg=1,interactive=False):
        print(deg)

        self.deg = deg

    def transform(self,X):
        if self.deg==1:
            return X
        elif self.deg==2:
            return torch.cat((X,self.transform2d(X)),dim=1)
        elif self.deg ==3:
            return torch.cat((X,self.transform2d(X),self.transform3d(X)),dim=1)
        else:
            return torch.cat((X,self.transform2d(X),self.transform3d(X),self.transform4d(X)),dim=1)



    def transform2d(self, X):
        d=X.shape[1]
        n = X.shape[0]
        new_X = []
        for idx in range(d):
            for jdx in range(idx+1,d):
                new_xi = X[:, idx] * X[:, jdx]
                new_X.append(new_xi)
        new_X = torch.stack(new_X).t()

        return new_X







    def transform3d(self, X):
        n = X.shape[0]
        d = X.shape[1]
        new_X=[]
        for idx in range(d):
            for jdx in range(idx+1, d):
                for kdx in range(jdx+1,d):
                    new_xi = X[:, idx] * X[:, jdx]  *X[:,kdx]
                    new_X.append(new_xi)
        new_X = torch.stack(new_X).t()

        return new_X

    def transform4d(self, X):
        n = X.shape[0]
        d = X.shape[1]

        new_X = []
        for idx in range(d):
            for jdx in range(idx+1, d):
                for kdx in range(jdx+1, d):
                    for ldx in range(kdx+1,d):
                        new_xi = X[:, idx] * X[:, jdx] * X[:, kdx] * X[:,ldx]
                        new_X.append(new_xi)

        new_X = torch.stack(new_X).t()
        return new_X



class RBFBasisTensor:

    def __init__(self,
                 k, std_from_clusters=True):

        self.k = k
        self.X = None
        self.std_from_clusters = std_from_clusters


    def rbf(self, X, c, s):
        distance = get_distance_tensor(X, c)
        return torch.exp(-distance / (s ** 2))



    def rbf_list(self, X, centroids, std_list):
        RBF_list = []
        rbf_list = torch.stack([self.rbf(X, c, s).flatten() for (c, s) in zip(centroids, std_list)])
        return rbf_list.t()

    def transform(self, X):

        if X.shape[0]==1 or self.X is not None:
            RBF_X = self.rbf_list(X, self.centroids, self.std_list)
            return RBF_X


        else:
            self.X = X

            self.centroids, self.std_list = kmeans(self.X.detach().cpu().numpy(), self.k, max_iters=2000)
            self.centroids = torch.tensor(np.array(self.centroids))
            self.std_list = torch.tensor(np.array(self.std_list))

            if not self.std_from_clusters:
                dMax = torch.max(torch.stack([get_distance_tensor(c1, c2) for c1 in self.centroids for c2 in self.centroids]))
                self.std_list = torch.repeat_interleave(dMax / torch.sqrt(2 * self.k), repeats=self.k)


            RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)
            return RBF_X


def uniform_grid(n_centers, low, high):
    """
    This function is used to create the parameters of uniformly spaced radial
    basis functions with 25% of overlap. It creates a uniformly spaced grid of
    ``n_centers[i]`` points in each ``ranges[i]``. Also returns a vector
    containing the appropriate scales of the radial basis functions.
    Args:
         n_centers (list): number of centers of each dimension;
         low (np.ndarray): lowest value for each dimension;
         high (np.ndarray): highest value for each dimension.
    Returns:
        The uniformly spaced grid and the scale vector.
    """
    n_features = low.shape[0]
    b = torch.zeros(n_features)
    c = list()
    tot_points = 1
    for i, n in enumerate(n_centers):
        start = low[i]
        end = high[i]

        b[i] = (end - start) ** 2 / n ** 3
        m = abs(start - end) / n
        if n == 1:
            c_i = (start + end) / 2.
            c.append(np.array([c_i]))
        else:
            c_i = np.linspace(start - m * .1, end + m * .1, n)
            c.append(c_i)
        tot_points *= n

    n_rows = 1
    n_cols = 0

    grid = torch.zeros((tot_points, n_features))

    for discrete_values in c:
        i1 = 0
        dim = len(discrete_values)

        for i in range(dim):
            for r in range(n_rows):
                idx_r = r + i * n_rows
                for c in range(n_cols):
                    grid[idx_r, c] = grid[r, c]
                grid[idx_r, n_cols] = discrete_values[i1]

            i1 += 1

        n_cols += 1
        n_rows *= len(discrete_values)

    return grid, b


class PolynomialBasisMushroom:
    #https: // github.com / MushroomRL / mushroom - rl / blob / dev / mushroom_rl / features / basis / polynomial.py
    r"""
    Class implementing polynomial basis functions. The value of the feature
    is computed using the formula:

    .. math::
        \prod X_i^{d_i}
    where X is the input and d is the vector of the exponents of the polynomial.
    """

    def __init__(self, dimensions=None, degrees=None):
        """
        Constructor. If both parameters are None, the constant feature is built.
        Args:
            dimensions (list, None): list of the dimensions of the input to be
                considered by the feature;
            degrees (list, None): list of the degrees of each dimension to be
                considered by the feature. It must match the number of elements
                of ``dimensions``.
        """
        self._dim = dimensions
        self._deg = degrees

        assert (self._dim is None and self._deg is None) or (
                len(self._dim) == len(self._deg))

    def __call__(self, x):

        if self._dim is None:
            return 1

        out = 1
        for i, d in zip(self._dim, self._deg):
            out *= x[i] ** d

        return out

    def __str__(self):
        if self._deg is None:
            return '1'

        name = ''
        for i, d in zip(self._dim, self._deg):
            name += 'x[' + str(i) + ']'
            if d > 1:
                name += '^' + str(d)
        return name

    @staticmethod
    def _compute_exponents(order, n_variables):
        """
        Find the exponents of a multivariate polynomial expression of order
        ``order`` and ``n_variables`` number of variables.
        Args:
            order (int): the maximum order of the polynomial;
            n_variables (int): the number of elements of the input vector.
        Yields:
            The current exponent of the polynomial.
        """
        pattern = np.zeros(n_variables, dtype=np.int32)
        for current_sum in range(1, order + 1):
            pattern[0] = current_sum
            yield pattern
            while pattern[-1] < current_sum:
                for i in range(2, n_variables + 1):
                    if 0 < pattern[n_variables - i]:
                        pattern[n_variables - i] -= 1
                        if 2 < i:
                            pattern[n_variables - i + 1] = 1 + pattern[-1]
                            pattern[-1] = 0
                        else:
                            pattern[-1] += 1
                        break
                yield pattern
            pattern[-1] = 0

    @staticmethod
    def generate(max_degree, input_size):
        """
        Factory method to build a polynomial of order ``max_degree`` based on
        the first ``input_size`` dimensions of the input.
        Args:
            max_degree (int): maximum degree of the polynomial;
            input_size (int): size of the input.
        Returns:
            The list of the generated polynomial basis functions.
        """
        assert (max_degree >= 0)
        assert (input_size > 0)

        basis_list = [PolynomialBasis()]

        for e in PolynomialBasisMushroom._compute_exponents(max_degree, input_size):
            dims = np.reshape(np.argwhere(e != 0), -1)
            degs = e[e != 0]

            basis_list.append(PolynomialBasisMushroom(dims, degs))

        return basis_list


class GaussianRBFTensor():
    def __init__(self,n_centers, low=None, high=None,env=None):
        self.n_centers = n_centers

        if low is not None and high is not None:

            self.low = torch.tensor(np.array(low))
            self.high = torch.tensor(np.array(high))
        elif env is not None:
            self.generate_scales(env)
        else:
            assert("Requires one of these - environment or low and high array")


        self.mean = []
        self.scale = []
        self.n_dim = self.low.shape[0]

        self.generate()

    def generate_scales(self,env):
        observation_examples = np.array([env.env.observation_space.sample() for x in range(10000)])
        min_val = np.min(observation_examples,axis=0)
        max_val = np.max(observation_examples,axis=0)
        self.low = min_val
        self.high = max_val


    def generate(self):
        n_features = len(self.low)
        assert len(self.n_centers) == n_features
        assert len(self.low) == len(self.high)

        grid, b = uniform_grid(self.n_centers, self.low, self.high)

        means = []
        for i in range(len(grid)):
            v = grid[i, :]
            means.append(v)

        self._mean, self._scale = torch.stack(means), b

    def transform(self,X):
        assert(X.shape[1]==self.n_dim)
        return torch.exp(-torch.sqrt(torch.sum(((X.reshape((-1,1,X.shape[-1])) - self._mean) ** 2)/self._scale.reshape(1,-1),dim=2)))




