
import torch.nn as nn
import sys
sys.path.append("experiments/")
from argparser_fqe import parse

args = parse()
import torch
use_cuda=False
torch.manual_seed(args.seed)
import random
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = DoubleTensor
random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)
import numpy as np
sys.path.append("domains_for_bilevel/interpretable_ope_public/")

sys.path.append("domains_for_bilevel/interpretable_ope_public/Cancer/")
import argparse
import pickle
sys.path.append("domains/hiv_domain/src/")
sys.path.append("domains/hiv_domain/hiv_domain/")
sys.path.append("domains/hiv_domain/")
sys.path.append("domains/")
sys.path.append("domains/Cancer/")

sys.path.append("src/")
from utils import *
import joblib
from collections import deque
from memory import *
from utils import *
from config import hiv_config

import torch.nn.functional as F

config = hiv_config

with open('domains/HIV/hiv_domain/hiv_simulator/hiv_preset_hidden_params', 'rb') as f:
    preset_hidden_params = pickle.load(f, encoding='latin1')

class TreePolicy():
    def __init__(self, eval_env, eps_behavior=0):
        self.eval_env= eval_env
        self.eps_behavior = eps_behavior

    def __call__(self, state, time_step=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

            # print(state.shape)
        if isinstance(state,np.ndarray)==False:
            state = state.detach().cpu().numpy()
        if state.shape[0] == 1:
            action, _=  self.eval_env.policy(state, eps=self.eps_behavior)
        else:
            action, _= self.eval_env.policy(state, eps=self.eps_behavior)

        return action


    def predict_proba(self, state, time_step=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

            # print(state.shape)
        if state.shape[0] == 1:
            action, prob =  self.eval_env.policy(state, eps=self.eps_behavior)
        else:
            action, prob = self.eval_env.policy(state, eps=self.eps_behavior)
        return prob


 # eval_qnet = QNet(config.state_dim,config.dqn_hidden_dims,config.action_size)
 #    load_qnet(eval_qnet,filename='qnet_mc.pth.tar')
 #    eval_qnet.eval()

class NNPolicy():
    def __init__(self, qnet, state_dim, num_actions, eps_behavior=0):
        self.qnet = qnet
        self.eps_behavior = eps_behavior
        self.num_actions = num_actions
        self.state_dim = state_dim

    def policy(self, state,eps = 0.05, transformed=False, normalized=False, scaler=None,flag=False):
        """Get the action under the current plan policy for the given state.

        Args:
        state: The array of state features


        Returns:
        The current greedy action under the planned policy for the given state. If no plan has been formed,
        return a random action.
        """
        softmax = nn.Softmax(dim=1)
        if transformed == True:
            if normalized == True:
                state = scaler.inverse_transform(state)

            features = self.qnet.forward2(state.type(DoubleTensor))
        else:
            features = self.qnet(
                state.type(DoubleTensor))
        if len(state.shape)==1:
            state= state.reshape(1,-1)
        if len(state.shape)==1 or state.shape[0]==1:
            if np.random.rand(1) < eps:
                assert(0)
                action = torch.tensor(np.random.randint(0, self.num_actions, state.shape[0])[0])
                probs = softmax_custom(features).reshape(self.num_actions)
                return action, probs
            else:
                probs = softmax_custom(
                    features).reshape(self.num_actions)
                return features.max(1)[1].view(1, 1), probs
        else:
            probs = softmax_custom(features).reshape(-1,self.num_actions)
            return features.max(1)[1].reshape(-1), probs


    def forward1(self, state):
        out = self.qnet.forward1(state)
        return out


    def forward2(self, state):
        out = self.qnet.forward2(state)
        return out


    def __call__(self, state, time_step=None,transformed=False, normalized=False, scaler=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            # print(state.shape)
        if isinstance(state, np.ndarray) ==True:
            state = torch.tensor(state)
        action, probs = self.policy(state, eps=self.eps_behavior,transformed=transformed,normalized=normalized,scaler=scaler)
        return action

    def predict_proba(self, state, transformed=False, normalized=False, scaler=None):
        """

        :param state: state features  matrix n x d
        :param transformed: Boolean True: if the state features are transformed using neural networks
        :param normalized:  True if the state features are scaled after transformation
        :param scaler: function used for scaling
        :return: action probabilities n x A
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        # get action probabilities for given state
        action, prob = self.policy(state, eps=self.eps_behavior, transformed=transformed, normalized=normalized,scaler=scaler)
        return prob

class Policy():
    def __init__(self, weights,num_actions):
        self.weights = weights
        if isinstance(self.weights,np.ndarray)==True:
            self.weights = torch.tensor(weights,requires_grad=True)

        self.num_actions = num_actions
    def __call__(self, states, time_step=None):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        d = states.shape[-1]

        if self.num_actions==2:
            values = torch.mm(self.weights.reshape(-1, 1), states)
        else:
            states = torch.repeat_interleave(states, repeats=self.num_actions, dim=0).reshape((-1, self.num_actions, d))
            values = states * self.weights.reshape((self.num_actions, d))
            values = torch.sum(values, dim=-1)
        actions = torch.argmax(values,dim=1)


        return actions


    def predict_proba(self, states, time_step=None):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        d = states.shape[-1]
        n = states.shape[0]
        if self.num_actions==2:
            probs = torch.zeros((n,2))
            values = torch.mm(states,self.weights.reshape(-1,1))
            sigmoid = torch.nn.Sigmoid()
            probs[:,1] = sigmoid(values + 1e-10).flatten()
            probs[:,0] = 1 -probs[:,1]

        else:
            states = torch.repeat_interleave(states,repeats=self.num_actions,dim=0).reshape((-1,self.num_actions,d))
            values = states* self.weights.reshape((self.num_actions,d))
            values = torch.sum(values + 1e-10,dim=-1)
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(values+1e-20)

        return probs

class SepsisPolicy():
    def __init__(self, eps_behavior=0,dim=46):
        self.model = DQN.load("data/sepsis_domain/deepq_sepsis")
        self.dim = dim
        self.eps_behavior = eps_behavior

    def __call__(self, state, time_step=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if (state.shape[1] > self.dim):
            state = state[:,:]
        state = state[:, 0:self.dim]

        state = state.reshape((-1,self.dim,1,1))
        action, _= self.model.predict(state)

        return action


    def predict_proba(self, state, time_step=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if (state.shape[1] > self.dim):
            state = state[:,:]
        state = state[:, 0:self.dim]

        state = state.reshape((-1,self.dim,1,1))
        action_probs  = self.model.action_probability(state)
        return action_probs


class HIVPolicy():
    def __init__(self, eps_behavior=0):
        self.eval_env = FittedQIteration(perturb_rate=0.05,
                                    preset_params=preset_hidden_params[config.ins],
                                    gamma=config.gamma,
                                    ins=config.ins,
                                    episode_length=config.max_length)
        self.eval_env.tree = joblib.load('domains/HIV/hiv_domain/hiv_domain_extra_tree_gamma_ins20.pkl')

        self.eps_behavior = eps_behavior

    def __call__(self, state, time_step=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

            # print(state.shape)
        if isinstance(state,np.ndarray)==False:
            state = state.detach().numpy()
        if state.shape[0] == 1:
            action, _=  self.eval_env.policy(state, eps=self.eps_behavior)
        else:
            action, _= self.eval_env.policy(state, eps=self.eps_behavior)

        return action


    def predict_proba(self, state, time_step=None):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

            # print(state.shape)
        if state.shape[0] == 1:
            action, prob=  self.eval_env.policy(state, eps=self.eps_behavior)
        else:
            action, prob = self.eval_env.policy(state, eps=self.eps_behavior)
        return prob




