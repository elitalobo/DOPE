# import Cancer

import os

from datetime import datetime
import sys
sys.path.append("experiments/")
sys.path.append("domains/")
sys.path.append("algorithms_ope/")
from argparser_fqe import parse
from cancer_env import *
from hiv_env import *
from sepsis_env import *
from cartpole_env import *
from custom import *
from mountaincar_env import *
from scipy.stats import gengamma
args = parse()
import torch

torch.manual_seed(args.seed)
import random

random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)
sys.path.append("../")

sys.path.append("src/")
from influence import *
from feature_expansion import *
from influence_utils import *
from influence_functions import *
from config import *
import copy
from dataloader import *
from fqe_method import *
from importance_sampling_method import *
from wdr_method import *
from utils_nn import *

from sklearn.preprocessing import PolynomialFeatures

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#print(sys.path)
# sys.path.append("/Users/elitalobo/PycharmProjects/untitled/soft-robust/interpretable_ope_public/")
sys.path.append("../domains_for_bilevel/")
# import Cancer


sys.path.append("algorithms_ope/")
from fqe_method import *
from wdr_method import *
from utils import *

import torch.nn as nn
sys.path.append("src/pytorch_influence_functions/")
sys.path.append("src/pytorch_influence_functions/pytorch_influence_functions/influence_functions/")
sys.path.append("src/pytorch_influence_functions/pytorch_influence_functions")
from hvp_grad import (
    calc_loss,
    s_test_sample_new,
    grad_z_new,
    s_test_cg,
)

class InfluenceAttack():

    def __init__(self, method_type="FQE", env=None, config=None, deg=3, sign=1, initial_lr=100.0,
                 type="l1", epsilon=0.1,  max_epochs=1000, reg=1e-2,eps=0.05, random=False, iters=100, attacker_type='influence',is_type="is",dataloader=None,frac=1.0, dataset_id=1):
        super(InfluenceAttack, self).__init__()
        """
        :param transitions: Array of transitions of size m1 consisting of tuples of the form [(s,a,r',s',time_step)]
        :param test_transitions: Array of transitions of size m2 consisting of tuples of the form [(s,a,r',s',time_step)]
        :param env: Instance of environment
        :param policy: evaluation policy -> Takes (state, time_step) as input and outputs an action id
        :param deg: Degree of polynomial features
        :param initial_lr: learning rate
        :sign: Direction of impact on value function
        :features: should be one of these -> ['rbf', 'poly', 'none']
        """
        # set no of iterations
        self.iters=iters
        # set method type
        self.method_type = method_type
        # set attacker type
        self.attacker_type = attacker_type
        self.influenceHelper = Influence()
        # set regularization parameter
        self.lamda = reg
        # set percentage of corrupt points
        self.alpha = eps
        # set sign of error
        self.sign = sign
        # set projection type
        self.projection_type = type
        # set regularization
        self.reg = reg

        # instance of environment
        self.env = env

        # set config
        self.config = config
        # set dataset_id
        self.dataset_id = dataset_id

        # Degree of polynomial features to be used in value function approximation
        self.deg = deg

        # random determines the type of random attack
        self.random = random

        # importance sampling type
        self.is_type = is_type

        self.max_epochs = max_epochs

        # initial learning rate for line search
        self.initial_lr = config.initial_lr

        # sign of error in value function estimate
        self.sign = config.sign
        #print("sign", self.sign)

        # discount factor
        self.gamma = self.env.discount_factor
        #print("self.env.discount", self.gamma)

        if dataloader is None:
            self.dataLoader = DataLoader( env, config,self.gamma,  num_samples=config.num_samples, num_trajectories=config.num_trajectories,type=type,frac=frac, dataset_id=self.dataset_id)
        else:
            self.dataLoader = dataloader
            # if frac is different, recalculate eps value.
            if frac != self.dataLoader.frac:
                self.dataLoader.frac = frac
                self.dataLoader.reset_eps()
        self.epsilon = self.dataLoader.eps

        # initialize OPE method
        self.initialize_methods()


    def initialize_methods(self):

        if self.method_type == 'FQE':
            # initial BRM method
            self.model = FQE_method(self.dataLoader,self.env,self.config,reg=self.reg)
        elif self.method_type == "IS":
            # initisalize IS method
            self.model = IS_method(self.dataLoader,self.env, self.config,is_type=self.is_type)
        else:
            # initial WDR method
            self.model = WDR_method(self.dataLoader,self.env, self.config,is_type=self.is_type)



    def get_most_influential_indices(self):
        """
        Computes the most influential indices for a given attack
        :return:
        """
        # get no of data points
        n=self.dataLoader.states.shape[0]


        #no of corrupt data points = no of train data points * percentage of corrupt data points
        num_corrupt = int(self.alpha * n)


        # if attacker is random:
        if self.attacker_type=='random':
            indices = np.arange(n)
            # shuffle indices
            np.random.shuffle(indices)
            #get random indices
            random_indices = indices[:num_corrupt]
            d=self.dataLoader.states.shape[-1]
            # get random updates
            updates = self.get_random_updates(d,num_corrupt,self.epsilon,self.projection_type)
            # get random updates
            updates = updates.detach().cpu().numpy()
            # get random indices, random  updates
            return random_indices, updates


        # append is_type for IS and WDR methods
        if self.method_type=="IS" or self.method_type=="WDR":
            method = self.method_type + "_" + self.is_type +"_"
        else:
            method = self.method_type + "_"
        try:
            # load influences for fsgm and influence attacks
            if self.attacker_type=='influence':
                influences = np.load("data/" + self.env.name + "/"  + self.attacker_type + "_" + str(self.dataset_id) + "_" +  method + str(self.sign*self.gamma) + "_influences.npy")
            elif self.attacker_type=='fsgm':
                influences = np.load("data/" + self.env.name + "/"  + self.attacker_type + "_" + str(self.dataset_id) + "_" +  method + str(self.sign*self.gamma) + "_gradients.npy")
                print("loaded")
            else:
                assert("Should not reach here!")
        except:

            # get total no of points
            n = self.dataLoader.states.shape[0]
            #get no of corrupt points

            num_corrupt = int(self.alpha * n)
            # Here we assume that the test error for fqe does not depend on influential data points
            test=True
            if self.method_type=="FQE":
                test=False
            try:
                if not os.path.exists("data/"+ self.env.name):
                    os.makedirs("data/"+ self.env.name)
            except:
                pass

            # if features are not transformed, use naive method for computing influence
            if self.config.transformed==False and self.attacker_type=='influence':
                influences = self.compute_all_influences_old(np.arange(n),test=test)
                #save influences
                np.save("data/" + self.env.name + "/"  + self.attacker_type + "_" + str(self.dataset_id) + "_" +  method + str(self.sign*self.gamma) + "_influences.npy",np.array(influences))
            # if features are transformed, use optimized approximation method to compute influence
            elif self.config.transformed==True and self.attacker_type=='influence':
                influences = self.compute_all_influences_new(np.arange(n),test=test)
                # influences = np.clip(influences,a_min=-1000,a_max=1000)
                # save influences
                np.save("data/" + self.env.name + "/"  + self.attacker_type + "_" + str(self.dataset_id) + "_" +  method + str(self.sign*self.gamma) + "_influences.npy",np.array(influences))

            elif self.attacker_type=='fsgm':
                # compute gradients for fsgm attack
                influences = self.compute_all_gradients(np.arange(n))
                # save gradients
                np.save("data/" + self.env.name + "/"  + self.attacker_type + "_" + str(self.dataset_id) + "_" +  method + str(self.sign*self.gamma) + "_gradients.npy",np.array(influences))

            else:
                assert("wrong attack_type")


            end = datetime.now()
            #print("time taken to compute all influence", (end-start).total_seconds())

        #Total influence of a data points = dual norm of the influence vector
        total_influence = compute_norm(influences,self.projection_type)
        #total_influence = np.sum(np.abs(influences),axis=1)

        #Sorts influence in decreasing order and gets indices of top num_corrupt data points
        influence_indices = np.argsort(-1.0 * total_influence).flatten()[:num_corrupt]
        #print("num_corrupt",num_corrupt)

        if self.random==1 and self.attacker_type=='influence':
            # For random ==1 , we select indices acccording to influence functions but choose perturbations randomly
            d=self.dataLoader.states.shape[-1]
            # compute random updates
            updates = self.get_random_updates(d,num_corrupt,self.epsilon,self.projection_type)
            updates = updates.detach().cpu().numpy()

            return influence_indices, updates
        elif self.random==2 and self.attacker_type=='influence':
            # For random==2 , we select indices randomly, but choose perturbations according to influence functions
            indices = np.arange(n)
            np.random.shuffle(indices)
            random_indices = indices[:num_corrupt]
            # set influence indices to random indices
            influence_indices = random_indices
            # get influences corresponding to random indices
            return influence_indices, influences[influence_indices]

        elif (self.attacker_type=='influence' or self.attacker_type=='fsgm') and (self.random==0 or self.random==3):
            # return indices of num_corrupt most influential points
            return influence_indices, influences[influence_indices]
        else:
            assert("shouldn't be here , wrong self.random")



    def recompute_influence_indices(self):
        """
        Recompute influences
        :return:
        """
        n=self.dataLoader.states.shape[0]
        test=True
        if self.method_type=="FQE":
            test=False

        #no of corrupt data points = no of train data points * percentage of corrupt data points
        num_corrupt = int(self.alpha * n)

        test=True
        influences = None
        if self.method_type=="FQE":
            test=False

            # if features are not transformed, use naive method for computing influence
            if self.config.transformed==False and self.attacker_type=='influence':
                influences = self.compute_all_influences_old(np.arange(n),test=test)
            # if features are transformed, use optimized approximation method to compute influence
            elif self.config.transformed==True and self.attacker_type=='influence':
                influences = self.compute_all_influences_new(np.arange(n),test=test)

        elif self.attacker_type=='fsgm':
            # compute gradients for fsgm attack
            influences = self.compute_all_gradients(np.arange(n))

        else:
            assert("wrong attack_type")

        # total influence = dual norm of the influence vector
        total_influence = compute_norm(influences,self.projection_type)
        #total_influence = np.sum(np.abs(influences),axis=1)

        #Sorts influence in decreasing order and gets indices of top num_corrupt data points
        influence_indices = np.argsort(-1.0 * total_influence).flatten()[:num_corrupt]

        return influence_indices


    def delete_and_recompute_returns(self, recompute=False):
        if recompute is True:
            influence_indices = self.recompute_influence_indices()
        else:
            influence_indices = self.influence_indices
        if self.method_type=="FQE":
            rets = self.model.delete_and_recompute_initial_returns(influence_indices)
        else:
            rets = self.model.get_initial_return()
        return rets





    def get_random_updates(self, d,num_corrupt,scale,type):
        """
        Based on Algorithm 4.1 in
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.556.2421&rep=rep1&type=pdf
        :param d:
        :param num_corrupt:
        :param scale:
        :param type:
        :return:
        """
        if type=="linf":
            points = np.random.uniform(-scale,scale,d*num_corrupt).reshape((num_corrupt,d))
            tensor_pts = torch.tensor(points)
            return tensor_pts
        if type=="l1":
            p=1
        elif type=="l2":
            p=2
        else:
            assert ("wrong type")
        assert(p!=0)
        # Generate nxd independent random real scalars ~ G(1/p,p)
        vals = gengamma.rvs(1.0/p, p, size=num_corrupt*d)
        signs = np.array([-1,1])
        # generate nxd random signs
        si =np.random.choice(signs,size=d*num_corrupt)
        # generate vector x = vals * signs
        x = vals * si

        x = x.reshape((num_corrupt,d))
        # generate n random variables uniformly distributed in [0,1]
        w = np.random.uniform(0,1,num_corrupt).reshape(-1,1)
        # set z = w^{1/d}
        z = np.power(w,1.0/d)
        # y = r * x * z / |x|_p
        points = scale*z*x/(np.linalg.norm(x,ord=p)+1e-20)
        points = torch.tensor(points)
        return points


    def attack(self,lim=1e-3):
        """

        :return: Calculates influence of each train data point and updates the train data points so that the test error
        increases
        """
        #print("attacking here")
        # get indices of the most influential set of points
        self.influence_indices, influences = self.get_most_influential_indices()
        influence_indices = self.influence_indices

        #Gets test error, and weights used for computing test error on the model
        test_error, weights_test, _ = self.model.get_test_error()
        prev_test_error = test_error

        self.iters=1

        for iter in range(self.iters):
            start_time = datetime.now()

            #Update previous test error

            start = datetime.now()
            #Recompute test error in each iteration since weights are updated
            test_error,  weights_test, l= self.model.get_test_error()
            end = datetime.now()
            #print("Time taken for getting test error", (end-start).total_seconds())

            #Break if no change in test error
            if torch.abs(prev_test_error - test_error) < lim and iter > 10:
                break

            start = datetime.now()
            #Recomputes train error in each iteration since weights and train dataset is updated
            train_error,xtrain, weights_train, v = self.model.get_train_error()
            end = datetime.now()
            #print("Time taken for getting train error", (end - start).total_seconds())


            #print("iter",iter)
            #print("test_error",test_error)
            #print("train error", torch.sum(train_error))

            if self.attacker_type=='influence':
                if (self.random==0 or self.random==2)==True:
                    projection = False
                else:
                    projection = True
                # get influence updates for the most influential data points
                #print("influence shape",influences.shape)
                # import ipdb; ipdb.set_trace()
                self.line_search(influences, influence_indices,projection=projection)
                prev_test_error = test_error
            elif self.attacker_type=='random':
                # get random updates for the most influential data points
                self.line_search(influences, influence_indices,projection=True)
                prev_test_error = test_error

            elif self.attacker_type=='fsgm':
                # fsgm attack is projected gradient descent attack
                self.line_search(influences, influence_indices,projection=True)
                prev_test_error = test_error


            else:

                assert("Attacker type is not valid. It can be only of two types - 'random' and 'influence'")


                del train_error, test_error, weights_train, weights_test, xtrain, v, l


            start = datetime.now()
            #Recomputes model weights
            #TODO Update weights
            self.model.reset_weights()

            end = datetime.now()
            #print("Time taken for recomputing model weights", (end - start).total_seconds())



            if self.attacker_type=="random" or self.attacker_type=='fsgm':
                break





    def update_features(self, new_param_values, indices, lr):
        """
        Updates next_state features of influential data points (m x state_dim) with new_param_values. The indices of the data points are stored in
        vector indices (m x 1)
        :param new_param_values: gradient of test error with respect to next-state features (m x state_dim)
        :param indices: indices of influential data points (m x1)
        :param lr: learning rate
        :return: new next_state features obtained by adding lr* new_param_value to existing next_state features
        """
        # get original next_state features
        original_features = self.dataLoader.get_state_features()
        original_features = original_features.detach().cpu().numpy()
        for index in range(len(new_param_values)):
            idx = indices[index]
            new_param_value = new_param_values[index]

            # update original features
            original_features[idx,:] = original_features[idx,:] + lr * new_param_value + 1e-8
        return original_features

    def save_model(self,train_error, test_error, isweights=None, cor_isweights=None):
        path  = "data/"  + self.method_type +"_" + self.attacker_type + "_" + self.env.name + "_" + str(self.epsilon) + "_" + str(self.alpha) + "_" + str(self.projection_type)  +"_"+ str(self.is_type) + "_" + str(self.dataset_id)
        if not os.path.exists(path):
            os.makedirs(path)

        weights = self.model.weights
        states = self.dataLoader.states
        original_states = self.dataLoader.actual_states
        actions = self.dataLoader.actions
        next_states = self.dataLoader.next_states
        influential_indices = self.influence_indices
        rewards = self.dataLoader.rewards

        if isinstance(weights,np.ndarray)==False:
            weights = weights.detach().cpu().numpy()

        if isinstance(states,np.ndarray)==False:
            states= states.detach().cpu().numpy()

        if isinstance(original_states,np.ndarray)==False:
            original_states = original_states.detach().cpu().numpy()

        if isinstance(actions,np.ndarray)==False:
            actions = actions.detach().cpu().numpy()

        if isinstance(next_states,np.ndarray)==False:
            next_states= next_states.detach().cpu().numpy()

        if isinstance(influential_indices,np.ndarray)==False:
            influential_indices = influential_indices.detach().cpu().numpy()

        if isinstance(weights,np.ndarray)==False:
            states= states.detach().cpu().numpy()

        if isinstance(rewards,np.ndarray)==False:
            rewards = rewards.detach().cpu().numpy()

        # if isinstance(train_error,np.ndarray)==False:
        #     train_error = train_error.detach().cpu().numpy()
        #
        # if isinstance(test_error,np.ndarray)==False:
        #     test_error = test_error.detach().cpu().numpy()


        # if isweights is not None and (isinstance(isweights,np.ndarray) is False):
        #     isweights = isweights.detach().cpu().numpy()
        #     cor_isweights = cor_isweights.detach().cpu().numpy()
        #     np.save(path + "/is_weights.npy",np.array(isweights))
        #     np.save(path + "/cor_weights.npy",np.array(cor_isweights))
        #
        #
        # #print("path",path)
        # np.save(path + "/weights.npy",weights)
        np.save(path + "/corrupt_states.npy",states)
        np.save(path + "/original_states.npy",original_states)
        # np.save(path + "/actions.npy", actions)
        # np.save(path + "/next_states.npy",next_states)
        np.save(path + "/train_error.npy",train_error)
        np.save(path + "/test_error.npy",test_error)
        np.save(path + "/influential_indices.npy",influential_indices)
        # np.save(path + "/rewards.npy",rewards)



    def check_constraint(self,new_features,indices):
        pass


    def frank_wolfe_update(self,influences,indices,lr):
        projection_center = self.dataLoader.actual_states.detach().cpu().numpy()[indices,:]

        delta = get_frank_wolfe_projection(influences, self.epsilon,self.projection_type)
        # delta =  projection_center - delta
        original_features =  self.dataLoader.actual_states.clone()
        original_features = original_features.detach().cpu().numpy()
        original_features[indices.flatten(),:] = original_features[indices.flatten(),:] + lr*delta
        return original_features

    def line_search(self, new_param_values, indices,projection=False):
        """
        Finds appropriate step size using line search and updates next_state features
        :param new_param_values: update for next_state features given by gradient of test error with respect
        to next_state features
        :param indices: indices of next_states that need to be updated
        :param threshold: threshold on train error
        :return: None
        """
        threshold = self.config.threshold
        if self.method_type=="IS":
            threshold = self.config.is_threshold
            threshold = np.inf
        # set lr to initial learning rate
        lr = self.initial_lr
        test_error, _, __ = self.model.get_test_error()
        train_error, _, __ , __ = self.model.get_train_error()
        print("test error before line search",test_error)

        # projection center are the original state features of the most influential data points
        projection_center = self.dataLoader.actual_states[indices]
        if isinstance(projection_center,np.ndarray)==False:
            projection_center = projection_center.detach().cpu().numpy()

        iter=0
        p=0.80
        flag=False
        while lr > 1e-4:
            #Obtain updated next_state features for learning rate lr

            if projection==True:
                new_s_features = self.update_features(new_param_values, indices, lr)
                project_features = new_s_features[indices]
                # get projected next_state action features
                projected_features = get_projected_features(project_features, projection_center,self.epsilon,self.projection_type)
                new_s_features[indices.flatten(),:] = projected_features

            else:

                new_s_features = self.frank_wolfe_update(new_param_values,indices,lr)

            # update next_state features of influential data points in the original next-state feature matrix

         # new next_state-action features
            s_features = torch.tensor(new_s_features)
            # get transformed new next-state features
            tr_s_features = s_features

            # construct state-action features with new next-state feature matrix

            # recompute q-value weights for evaluation policy using train data
            weights = self.model.get_weights(s_features=tr_s_features)

            #get new test error
            new_test_error, _, __ = self.model.get_test_error(weights=weights,states=tr_s_features)

            # get new train error
            new_train_error, _, __, pbe = self.model.get_train_error(weights=weights,s_features=tr_s_features,grad=False)

            print("intermediate test error",new_test_error)
            print("intermediate new train",new_train_error)
            print("threshold", threshold)


            if self.attacker_type=='random':
                if (new_train_error <= threshold):
                    flag=True
                    break

            if (new_test_error < test_error) or (new_train_error > threshold):
                print("new test error", new_test_error)
                print("train error",new_train_error)
                lr = lr * p

            else:
                flag=True
                break

            iter+=1

        print("new test error", new_test_error)
        print("new train error", new_train_error)
        print("old test error", test_error)
        print("Final lr", lr)
        print("projected features diff",np.sum(np.abs(projection_center-new_s_features[indices.flatten(),:])))
        # update next_state features
        if flag==True:
            #print("here")
            self.dataLoader.set_state(new_s_features)
            self.model.reset_weights()



    def compute_all_influences_new(self,indices, test=True):
        """

        :param test_error:
        :param train_error:
        :param param:
        :param xtrain_arr:
        :return:
        """


        start = datetime.now()

        r=10
        gtest_hess = s_test_sample_new(
            self.model,
            self.dataLoader,
            gpu=False,
            damp=0.0,
            r=r,
            recursion_depth=int(self.dataLoader.states.shape[0]/r), #10000
        )
        print(int(self.dataLoader.states.shape[0]/r))
        gtest_hess = gtest_hess[0].t()

        train_error, xtrain, w_train, l = self.model.get_train_error( grad=True,
                                                     indices=indices)
        jac = torch.autograd.grad(train_error, w_train, create_graph=True, retain_graph=True)[0]
        elem_prod = torch.matmul(gtest_hess.reshape(1, -1), jac.reshape(-1, 1))

        influences = -1.0 * torch.autograd.grad(elem_prod, xtrain)[0].detach().cpu().numpy()
        print("influences", np.sum(influences))

        if test == True:
            test_error, w_test, xtest = self.model.get_test_error()

            test_grad = torch.autograd.grad(test_error, xtest, retain_graph=True)[0].detach().cpu().numpy()
            if test_grad.shape[0] != influences.shape[0]:
                test_grad = test_grad[indices.flatten(), :]
            influences = influences + test_grad

            del test_grad
        del train_error, xtrain, w_train

        influences = np.array(influences)

        end = datetime.now()
        #print("Time taken to get influences", (start - end).total_seconds())

        return np.array(influences)


    def compute_all_gradients(self,indices):
        """

        :param test_error:
        :param train_error:
        :param param:
        :param xtrain_arr:
        :return:
        """

        def train_f(weights, single=False, grad=False):
            loss, __, ___, _ = self.model.get_train_error(weights=weights, grad=grad)
            return loss

        weights = self.model.get_weights()

        if isinstance(weights,np.ndarray)==True:
            weights = torch.tensor(weights)

        train_error, xtrain, w_train, l = self.model.get_train_error( grad=True,
                                                                      indices=indices)


        start = datetime.now()
        """ Compute jacobian of test error with respect to the weights_test """
        gtrain =  torch.autograd.grad(train_error, xtrain, create_graph=True, retain_graph=True)[0]
        #print("gtest",torch.sum(gtest))
        gtrain = gtrain.detach().cpu().numpy()

        influences = gtrain
        return influences


    def compute_all_influences_old(self,indices, test=True,hessian=None):
        """
        :param test_error:
        :param train_error:
        :param param:
        :param xtrain_arr:
        :return:
        """

        def test_f(weights=None):

            loss, w, __ = self.model.get_test_error(weights=weights)
            return loss

        def train_f(weights, single=False, grad=False):
            loss, __, ___, _ = self.model.get_train_error(weights=weights, grad=grad)
            return loss

        weights = self.model.get_weights()

        if isinstance(weights,np.ndarray)==True:
            weights = torch.tensor(weights)


        start = datetime.now()
        """ Compute jacobian of test error with respect to the weights_test """
        gtest = torch.autograd.functional.jacobian(test_f, weights.detach().flatten().requires_grad_(True))
        #print("gtest",torch.sum(gtest))
        if hessian is None:
            hessian_val = torch.autograd.functional.hessian(train_f, weights.detach().flatten())
        else:
            #print("got hessian")
            hessian_val = hessian


        end = datetime.now()
        #print("Time taken to compute hessian", (start - end).total_seconds())

        # #print("rank", np.linalg.matrix_rank(hessian_val.detach().cpu().numpy()))
        """ Computes inverse of hessian of train error """

        hessian_val = hessian_val.detach().cpu().numpy()
        gtest = gtest.detach().cpu().numpy()

        gtest_hess = get_hessian_inverse_vector_product(gtest, hessian_val)
        gtest_hess = torch.tensor(gtest_hess.astype(np.double))
        end = datetime.now()
        #print("time taken via backpropagation", (end-start).total_seconds())

        start = datetime.now()

        train_error, xtrain, w_train, l = self.model.get_train_error(weights=weights.detach().requires_grad_(True), grad=True)
        jac = torch.autograd.grad(train_error,w_train, create_graph=True,retain_graph=True)[0]
        elem_prod = torch.matmul(gtest_hess.reshape(1, -1), jac.reshape(-1, 1))

        influences = -1.0 * torch.autograd.grad(elem_prod, xtrain)[0].detach().cpu().numpy()
        print("influences", np.sum(influences))

        if test==True:
            #print("test is true")
            test_error, w_test, xtest = self.model.get_test_error(weights=weights.detach().requires_grad_(False))

            test_grad = torch.autograd.grad(test_error,xtest,retain_graph=True)[0].detach().cpu().numpy()
            if test_grad.shape[0] != influences.shape[0]:
                test_grad = test_grad[indices.flatten(),:]
            influences = influences + test_grad

            del test_grad
        del train_error, xtrain, w_train
        return influences



def get_hessian_inverse_vector_product(v,hessian):
    Q, R = np.linalg.qr(hessian)
    Q_inv = Q.transpose()
    R_inv = np.linalg.inv(R)
    temp = np.matmul(v.reshape(1,-1),R_inv)
    hv = np.matmul(temp.reshape(1,-1),Q_inv)
    return hv
