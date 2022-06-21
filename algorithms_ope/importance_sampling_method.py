import os

from datetime import datetime
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
sys.path.append("../")

sys.path.append("src/")
from influence import *
from feature_expansion import *
from influence_utils import *
from influence_functions import *
import copy
from dataloader import *
from sklearn.linear_model import LinearRegression, LogisticRegression
sys.path.append("../domains_for_bilevel/")

from sklearn.preprocessing import PolynomialFeatures

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(sys.path)
# sys.path.append("/Users/elitalobo/PycharmProjects/untitled/soft-robust/interpretable_ope_public/")
sys.path.append("../domains_for_bilevel/")
# import Cancer


from utils import *

import torch.nn as nn


class IS_method(nn.Module):
    def __init__(self, dataloader,env, config,is_type="is", reg=1e-3, max_epochs=5000,clipped=0.05):
        """
        :param dataloader: None if it is a new expereiment
        :param env:
        :param config: environment config
        :param is_type: importance sampling type
        :param reg: regularization parameter
        :param max_epochs: maximum no of epochs for learning behavior policy
        :param clipped: clipped parameter
        """
        super(IS_method,self).__init__()
        self.lamda = reg # regulariation parameter
        self.env = env
        self.config = config
        self.max_epochs = max_epochs
        self.sign = config.sign
        self.clipped = clipped



        self.dataloader = dataloader

        self.gamma = config.gamma
        self.is_operator=None

        #initialize is operator
        if is_type=="cpdis":
            self.is_operator = self.get_consistent_pdis_estimate
        elif is_type=="pdis":
            self.is_operator = self.get_pdis_estimate
        elif is_type=='wpdis':
            self.is_operator = self.get_wpdis_estimate
        else:
            self.is_operator = self.get_is_estimate

        # evaluation policy
        self.policy_eval = self.dataloader.policy_eval

        # Estimate initial behavior policy weights
        self.reset_weights()
        # Construct behavior policy
        self.policy_beh = Policy(self.weights, num_actions=self.config.action_size)

        # is weight
        self.is_weights = None

        # influence indices
        self.influence_indices=None

    def compute_train_loss(self,indices):
        """
        :param indices: indices to construct train loss for
        :return: mean of train loss for the given indices
        """
        train_error, states, w, v_error =  self.get_train_error(s_features=None, grad=True, indices=indices)
        return train_error

    def compute_test_loss(self):
        """
        computes test loss and returns it.
        :return:
        """
        total_error,_, __ = self.get_test_error()
        return total_error



    def estimate_behavior_policy(self, states=None, actions=None, percent=0.8):
        """
        https://github.com/elitalobo/OPE-tools/blob/master/ope/algos/fqe.py
        :param max_epochs:
        :param epsilon:
        :return:
        """
        # get state features
        if states is None:
            states = self.dataloader.states
        # get action features
        if actions is None:
            actions = self.dataloader.actions

        # use x % of the data to learn the behavior policy
        num_samples = int(percent * states.shape[0])

        # get test states
        test_states = self.dataloader.states[num_samples:, :].detach().cpu().numpy()
        # get test actions
        test_actions = self.dataloader.actions.flatten()[num_samples:].detach().cpu().numpy()

        # train states
        train_states = states.detach().cpu().numpy()
        # train actions
        train_actions = actions.detach().cpu().numpy()
        # learn logistic regression model
        # training using train states and train actions
        clf = LogisticRegression(C=1.0 / self.lamda, fit_intercept=False, max_iter=self.max_epochs,
                                 multi_class='multinomial').fit(train_states,
                                                                train_actions.flatten())
        # Predicting accuracy on test samples
        print("Accuracy", clf.score(test_states, test_actions))

        # returning weights of the logistic regression
        return clf.coef_.flatten()

    def get_weights(self, s_features=None, actions=None, policy=None):
        """
            Computes weights of q-value function of policy pi from train data
            :return: weights  (state_dim |a|,)
        """
        # estimate weights of behavior policy
        weights = self.estimate_behavior_policy(states=s_features, actions=actions)
        return weights

    def reset_weights(self, s_features=None, actions=None, policy=None):
        """
            Computes weights of q-value function of policy pi from train data
            :return: weights  (state_dim |a|,)
        """
        # estimate weights of behavior policy
        weights = self.estimate_behavior_policy(states=s_features,actions=actions)
        print("weights sum", np.sum(np.abs(weights)))
        # convert weights to tensor weights
        weights = torch.tensor(weights,requires_grad=True)
        # Wrap weights in nn.Parameter to add it to the model
        self.weights = torch.nn.Parameter(weights)


    def get_test_error(self, weights=None, states=None, tr_states=None, debug=False):
        """
        Computes test loss
        :param weights: weights for the behavior policy
        :param grad: Indicates if requires_grad should be set true for weights
        :param states: state features without transformation
        :param tr_states: state features with transformation
        :param debug: debug=True indicates that the is_weights, policy weights etc should be printed
        :return:
        """
        # set weight if weight is none
        if weights is None:
            weights = self.weights.reshape(-1, 1)

        if isinstance(weights, np.ndarray) == True:
            weights = torch.tensor(weights).reshape(-1, 1)

        # initialize behavior policy
        policy_beh = Policy(weights=weights, num_actions=self.config.action_size)

        # get expected returns and differentiable state features from is_operator
        val, _, states = self.is_operator(policy_beh, states=states, tr_states=tr_states, debug=debug)
        error = self.sign * val
        return error, weights, states

    def get_is_estimate(self, policy_beh, states=None, tr_states=None, debug=False):
        """
        Computes plain importance sampling with weight normalization
        :param policy_beh:
        :param states:
        :param tr_states:
        :param debug:
        :return:
        """
        #number of trajectories
        num_traj = len(self.dataloader.traj_set.trajectories)
        # matrix of values of each trajectory
        value = torch.zeros(num_traj)
        #weights of trajectory
        weights = torch.zeros(num_traj)
        # get actions from dataloader
        actions = torch.tensor(self.dataloader.actions).long()
        if states is None:
            # get state features
            states = self.dataloader.get_state_features(requires_grad=True)

        if isinstance(states, np.ndarray) == True:
            states = torch.tensor(states, requires_grad=True)

        n = states.shape[0]

        # # Compute evaluation policy probabilities of given states
        eval_probs = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)[
            np.arange(n), actions.flatten()]
        # get eval probs for states
        # eval_probabilities = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)
        # # compute log of eval probabilities
        # log_eval_probs = torch.log(eval_probabilities+ 1e-10)
        # # get gumbel softmax probabilities
        # gumbel_probs = torch.nn.functional.gumbel_softmax(log_eval_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)

        # extract probabilities of actions selected
        # eval_probs = gumbel_probs[
        #     np.arange(n), actions.flatten()]


        # Compute behavior policy on transformed states
        beh_probabilities = policy_beh.predict_proba(states)[
            np.arange(n), actions.flatten()]
        beh_probs = beh_probabilities

        index = 0
        # iterate over all trajectories
        for i_traj in range(num_traj):
            # get transitions from i^{th} trajectory
            transitions = self.dataloader.traj_set.trajectories[i_traj].transitions
            l = len(transitions)
            tmp = torch.ones(1)[0]
            # iterate over all transitions
            for n in range(l):
                # multiply tmp by eval_prob(s)/beh_prob(state)
                tmp = tmp * (eval_probs[index])
                tmp = tmp / (beh_probs[index] + self.clipped)
                index += 1
            # is weight of the ith trajectory
            weights[i_traj] = tmp

        if debug == True:
            num_near_zero = torch.sum(beh_probs <= 0.01 / self.config.action_size)
            print("weight sum", torch.sum(weights))
            print("sum probs", torch.sum(beh_probs))
            print("policy weights ", np.sum(np.abs(self.weights)))
            print("num_near_zero", num_near_zero)
            print("is weights sum", torch.sum(weights))

        # if self.is_weights is None:
        self.is_weights = weights

        # if self.wis == True:
        # normalize the weights
        norm_weights = (weights) / (torch.sum(weights) + 1e-10)

        for i_traj in range(num_traj):
            # multiple probability of each trajectory with corresponding discounted returns
            value[i_traj] = self.dataloader.traj_set.trajectories[i_traj].disc_rets * norm_weights[i_traj]
        # sum contributions of trajectory
        val = torch.sum(value)
        print(torch.sum(weights))
        print("val", val)
        importance_weights = weights
        # return expected return, importance weights and differentiable state features
        return val, importance_weights, states

    def get_wpdis_estimate(self, policy_beh, states=None, tr_states=None, debug=False):
        """
        Computes weighted per-decision importance sampling estimate
        :param policy_beh:
        :param states:
        :param tr_states:
        :param debug:
        :return:
        """
        # get lower bound on returns
        lb = self.dataloader.rmin * (1 - np.power(self.gamma, self.dataloader.max_traj_length)) / (1 - self.gamma)
        # get upper-bound on returns
        ub = self.dataloader.rmax * (1 - np.power(self.gamma, self.dataloader.max_traj_length)) / (1 - self.gamma)
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        # get number of trajectories
        num_traj = len(self.dataloader.traj_set.trajectories)

        if states is None:
            # get states if not provided
            states = self.dataloader.get_state_features(requires_grad=True)

        actions = torch.tensor(self.dataloader.actions).long()
        if isinstance(states, np.ndarray):
            states = torch.tensor(states)
        n = states.shape[0]
        d = states.shape[1]

        # Get evaluation probabilities for rescaled states
        eval_probabilities = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)
        # compute log of eval probabilities
        # log_eval_probs = torch.log(eval_probabilities+ 1e-10)
        # # compute gumbel probs of log eval probabilities
        # gumbel_probs = torch.nn.functional.gumbel_softmax(log_eval_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)
        # # extract probabilities of selected actions
        # eval_probs = gumbel_probs[
        #     np.arange(n), actions.flatten()]

        # get eval probabilities
        eval_probs = eval_probabilities[
            np.arange(n), actions.flatten()]

        # get behavior probability for transformed states
        beh_probs = policy_beh.predict_proba(states)[
            np.arange(n), actions.flatten()]

        index = 0
        rewards = torch.tensor(self.dataloader.rewards)
        traj_contributions = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        # initialize is weights
        weights = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        # iterate over trajectories
        for i_traj in range(num_traj):
            transitions = self.dataloader.traj_set.trajectories[i_traj].transitions
            l = len(transitions)
            is_weight = torch.ones(1)[0]
            discount = torch.ones(1)[0]
            # iterate over transitions
            for n in range(l):
                # multiple is weight by pi_e(s,a)/(pi_b(s,a) + self.clipped)
                is_weight = is_weight * eval_probs[index]
                is_weight = is_weight / (beh_probs[index] + self.clipped)
                # compute per-decision step discounted rewards
                traj_contributions[i_traj, n] = discount * is_weight * rewards[index]
                weights[i_traj, n] = is_weight
                # multiply discount by gamma for the next time step
                discount = discount * self.gamma
                index = index + 1

        weight_sum = torch.sum(weights)
        # if self.is_weights is None:
        self.is_weights = weights
        print("got is weights")

        val = 1.0 / (ub - lb) * (torch.mean(torch.sum(traj_contributions, dim=1)) - lb)
        # take mean value of trajectories
        # val = torch.mean(torch.sum(traj_contributions, dim=1))
        #TODO Check this
        val = torch.sum(traj_contributions)/(weight_sum + 1e-10)
        num_near_zero = torch.sum(beh_probs <= 1e-2)
        # if states is None:
        print("sum probs", torch.sum(beh_probs))

        # print("policy weights ", np.sum(np.abs(self.weights)))
        print("num_near_zero", num_near_zero)
        print("is weights sum", torch.sum(weights))
        print("val", val)
        importance_weights = weights

        return val, importance_weights, states



    def get_pdis_estimate(self, policy_beh, states=None, tr_states=None, debug=False):
        """

        :param policy_beh:
        :param states:
        :param tr_states:
        :param debug:
        :return:
        """

        lb = self.dataloader.rmin * (1 - np.power(self.gamma, self.dataloader.max_traj_length)) / (1 - self.gamma)
        ub = self.dataloader.rmax * (1 - np.power(self.gamma, self.dataloader.max_traj_length)) / (1 - self.gamma)
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        num_traj = len(self.dataloader.traj_set.trajectories)

        if states is None:
            states = self.dataloader.get_state_features(requires_grad=True)

        actions = torch.tensor(self.dataloader.actions).long()
        if isinstance(states, np.ndarray):
            states = torch.tensor(states)
        n = states.shape[0]
        d = states.shape[1]

        # # Get evaluation probabilities for rescaled states
        eval_probs = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)[
            np.arange(n), actions.flatten()]

        # # get eval probabilities
        # eval_probabilities = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)
        # # compute log of eval probabilities
        # log_eval_probs = torch.log(eval_probabilities+ 1e-10)
        # # compute gumbel probs from log of eval probabilities
        # gumbel_probs = torch.nn.functional.gumbel_softmax(log_eval_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)
        # # extract probabilities of selected actions
        # eval_probs = gumbel_probs[
        #     np.arange(n), actions.flatten()]


        # get behavior probability for transformed states
        beh_probs = policy_beh.predict_proba(states)[
            np.arange(n), actions.flatten()]

        index = 0
        rewards = torch.tensor(self.dataloader.rewards)
        traj_contributions = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        weights = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        # iterate over trajectories
        for i_traj in range(num_traj):
            transitions = self.dataloader.traj_set.trajectories[i_traj].transitions
            l = len(transitions)
            is_weight = torch.ones(1)[0]
            discount = torch.ones(1)[0]
            # iterate over transitions
            for n in range(l):
                # multiple is_weight by (pi_e(s,a)/pi_b(s,a))
                is_weight = is_weight * eval_probs[index]
                is_weight = is_weight / (beh_probs[index] + self.clipped)
                # compute per-decision step discounted rewards
                traj_contributions[i_traj, n] = discount * is_weight * rewards[index]
                weights[i_traj, n] = is_weight
                # multiply discount by gamma for the next time step
                discount = discount * self.gamma
                index = index + 1

        weight_sum = torch.sum(weights)
        # if self.is_weights is None:
        self.is_weights = weights
        print("got is weights")

        # val = 1.0 / (ub - lb) * (torch.mean(torch.sum(traj_contributions, dim=1)) - lb)
        # take mean value of trajectories
        val = torch.mean(torch.sum(traj_contributions, dim=1))
        # val = torch.sum(traj_contributions)/(weight_sum + 1e-10)
        num_near_zero = torch.sum(beh_probs <= 1e-2)
        # if states is None:
        print("sum probs", torch.sum(beh_probs))

        # print("policy weights ", np.sum(np.abs(self.weights)))
        print("num_near_zero", num_near_zero)
        print("is weights sum", torch.sum(weights))
        print("val", val)
        importance_weights = weights

        return val, importance_weights, states


    def get_consistent_pdis_estimate(self, policy_beh, states=None, tr_states=None, debug=False):
        """
        Computes consistent PDIS estimate
        :param policy_beh:
        :param states:
        :param tr_states:
        :param debug:
        :return:
        """
        lb = self.dataloader.rmin * (1 - np.power(self.gamma, self.dataloader.max_traj_length)) / (1 - self.gamma)
        ub = self.dataloader.rmax * (1 - np.power(self.gamma, self.dataloader.max_traj_length)) / (1 - self.gamma)
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        num_traj = len(self.dataloader.traj_set.trajectories)

        if states is None:
            states = self.dataloader.get_state_features(requires_grad=True)

        actions = torch.tensor(self.dataloader.actions).long()

        if isinstance(states, np.ndarray):
            states = torch.tensor(states)
        n = states.shape[0]
        d = states.shape[1]
        # get eval probabilities
        # eval_probabilities = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)
        # get log of eval probabilities
        # log_eval_probs = torch.log(eval_probabilities+ 1e-10)
        # # get gumbel of log probabilities
        # gumbel_probs = torch.nn.functional.gumbel_softmax(log_eval_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)
        # # extract probabilities of selected actions
        # eval_probs = gumbel_probs[
        #     np.arange(n), actions.flatten()]
        # get eval probabilities
        eval_probs = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)[
            np.arange(n), actions.flatten()]
        # get beh probabilities
        beh_probs = policy_beh.predict_proba(states)[
            np.arange(n), actions.flatten()]

        index = 0
        # get rewards
        rewards = torch.tensor(self.dataloader.rewards)
        # initialize trajectory contributions to zero
        traj_contributions = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        # initialize is weights
        weights = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        for i_traj in range(num_traj):
            # get transitions of the i^{th} trajectory
            transitions = self.dataloader.traj_set.trajectories[i_traj].transitions
            l = len(transitions)
            tmp = torch.ones(1)[0]
            is_weight = torch.ones(1)[0]
            discount = torch.ones(1)[0]

            for n in range(l):
                # state = transitions[n].state
                # multiply is weights by pi_e(s,a)/pi_b(s,a)
                is_weight = is_weight * eval_probs[index]
                is_weight = is_weight / (beh_probs[index] + self.clipped)
                # traj[i,:n] contribution = \gamma^n \rho_{i,:n} * reward
                traj_contributions[i_traj, n] = discount * is_weight * rewards[index]
                weights[i_traj, n] = is_weight
                # multiply discount factor by gamma
                discount = discount * self.gamma
                index = index + 1
        # divide weight of each time step by average weight for that time step
        traj_cont = traj_contributions / (torch.sum(weights, 0).reshape(1, -1) + 1e-10)

        # if self.is_weights is None:
        self.is_weights = weights

        val = 1.0 / (ub - lb) * (torch.sum(traj_contributions) - lb)
        # sum contributions of trajectories
        val = torch.sum(traj_cont)
        if debug == True:
            num_near_zero = torch.sum(eval_probs <= self.clipped)
            print("sum probs", torch.sum(beh_probs))
            print("policy weights ", np.sum(np.abs(self.weights)))
            print("num_near_zero", num_near_zero)
            print("is weights sum", torch.sum(weights))
        importance_weights = weights

        return val, importance_weights, states

    def get_initial_return(self, weights=None, policy=None):
        """
        Evaluates initial returns
        :param weights: weights of the q-value function of policy policy_eval
        :param policy_weights:
        :return:
        """
        if weights is None:
            w = torch.tensor(self.weights, requires_grad=True).reshape(-1, 1).double()
        elif isinstance(weights,np.ndarray)==True:
            w = torch.tensor(weights, requires_grad=True).reshape(-1, 1).double()
        else:
            w= weights.reshape(-1, 1).double()

        policy_beh = Policy(w, num_actions=self.config.action_size)

        # get expected returns from is_operator
        val, _, test_states = self.is_operator(policy_beh)
        print("val")

        return val.detach().cpu().numpy()

    def get_train_error(self, weights=None, s_features=None, grad=True,indices=None):
        """
        Computes train error using weights of q-value function of evaluation policy
        :param weights: weights of q-value function (state_dim x 1)
        :return:
        """
        if indices is not None:
            n = self.dataloader.states.shape[0]
            # get states and  transformed states for indices
            states = self.dataloader.get_state_features_indices(requires_grad=grad, indices=indices)
            # get actions
            actions = self.dataloader.actions[indices.flatten()]


        elif s_features is None and grad == True:
            # get all states and transformed states
            states = self.dataloader.get_state_features(requires_grad=grad)
            actions = self.dataloader.actions

        else:
            # No need to transform features here
            if s_features is None:
                states = self.dataloader.states
            else:
                states = s_features
            if isinstance(s_features, np.ndarray) == True:
                # Assuming we got transformed features
                states = torch.tensor(s_features, requires_grad=grad)
            actions = self.dataloader.actions

        if weights is None:
            w = self.weights.reshape(-1, 1)
        else:
            w = weights
            if isinstance(w, np.ndarray) is True:
                w = torch.tensor(w, requires_grad=True)
            w = w.reshape(-1, 1)

        policy_beh = Policy(w, num_actions=self.config.action_size)

        # train error is the negative loglikelihood of all actions
        log_likelihood = torch.sum(
            torch.log(policy_beh.predict_proba(states)[np.arange(states.shape[0]), actions.flatten()] + 1e-20))

        n = self.dataloader.states.shape[0]
        n1 = states.shape[0]
        error = -log_likelihood + self.lamda * torch.sum(w ** 2) *(n1/n)
        # print("error", -log_likelihood)

        # print("train error here", -log_likelihood)

        # TODO: Assert that gtest, hessian and mixed derivative are non-zero, change double to double
        return error, states, w, -log_likelihood



