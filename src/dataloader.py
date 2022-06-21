import os
import sys
import sys
sys.path.append("experiments/")
#from argparser_fqe import parse
sys.path.append("src/")
from argparser_fqe import *
args = parse()
import torch

torch.manual_seed(args.seed)
import random

random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)
from feature_expansions import *
from feature_expansions import GaussianRBFTensor
import copy
from create_datasets import *
from utils import Transition



class ProcessedTrajectory:
    """
    https://github.com/dtak/interpretable ope public.git
    """
    def __init__(self, trajectory, gamma):
        self.transitions = []
        len=0
        for transition in trajectory:
            state = transition.state
            next_state = transition.next_state
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            self.transitions.append(Transition(state,transition.action,
                                               next_state,transition.reward,transition.done,None))
            len+=1

        self.gamma = gamma
        self.length = len
        self.disc_rets = self.return_trajectory_return(self.gamma)

    def add_transition(self, trans):
        self.transitions.append(trans)


    def __repr__(self):
        final_repr = ""

        for trans in self.transitions:
            final_repr += repr(trans) + "\n"
        return final_repr

    def return_trajectory_return(self, gamma):
        traj_return = 0
        discout = 1
        for trans in self.transitions:
            traj_return += trans.reward * discout
            discout *= gamma
        self.trajectory_return = traj_return

        return traj_return

    def return_traj_len(self):
        return len(self.transitions)

    def return_initial_state(self):
        return self.transitions[0].state

    def return_states_extremes(self):
        min_states_values = self.transitions[0].state
        max_states_values = self.transitions[0].state
        for trans in self.transitions:
            state = trans.state
            next_state = trans.next_state
            min_states_values = np.minimum(min_states_values, state)
            max_states_values = np.maximum(max_states_values, state)
            min_states_values = np.minimum(min_states_values, next_state)
            max_states_values = np.maximum(max_states_values, next_state)
        return min_states_values, max_states_values


class TrajectoriesDataSet:
    """
    https://github.com/dtak/interpretable ope public.git
    """
    def __init__(self):
        self.trajectories = []

    def add_trajectory(self, trajectory):
        assert isinstance(trajectory, ProcessedTrajectory), (
            "Error in TrajectoriesDataSet.add_trajectory(): "
            "Input must be a Transition")
        self.trajectories.append(trajectory)

    def remove_trajectory(self, idx_of_trajectory_to_remove):
        del self.trajectories[idx_of_trajectory_to_remove]

    def return_num_traj(self):
        return len(self.trajectories)

    def return_as_TransitionsDataSet(self):
        trans_dataset = TransitionsDataSet()
        for traj in self.trajectories:
            trans_dataset.add_trajectory(traj)
        return trans_dataset

import joblib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from sklearn.preprocessing import StandardScaler


class DataLoader():
    def __init__(self, env, global_config, gamma, type="l1",frac=1.0,num_samples=20000000,distill=False, num_trajectories=None, dataset_id=1):
        # Extracts states, actions, next_states, rewards from set of transitions

        self.gamma = gamma

        self.num_samples = num_samples

        self.env = env
        self.type = type

        self.scaler=None
        self.eps=None
        self.frac = frac
        self.dataset_id = dataset_id
        print(self.dataset_id)


        self.global_config = global_config

        # self.config, self.trajectories, self.test_trajectories, self.policy_eval, self.policy_beh = generate_datasets(env,global_config,num_trajectories,gamma=env.discount_factor,args=None)

        self.load_config()
        self.load_datasets()
        self.load_policies()
        self.extract_data()
        self.traj_set, self.train_rets, max_len = self.process_trajectories(self.trajectories)
        self.test_traj_set, self.test_rets, max_test_len = self.process_trajectories(self.test_trajectories)
        self.eval_traj_set, self.eval_rets, max_eval_len = self.process_trajectories(self.eval_trajectories)
        print("eval rets",self.eval_rets)
        self. max_traj_length = max_len

        self.rmax = torch.max(self.rewards)
        self.rmin = torch.min(self.rewards)
        print("states shape",self.states.shape)


    def load_config(self, dir="datasets/"):
        self.config = joblib.load(dir + self.env.name + "_config.pkl")

    def load_datasets(self,  dir="datasets/"):
        type="train"
        self.trajectories = joblib.load(dir + self.env.name +  "_" + type + "_"+ str(self.dataset_id) + "_transitions.pkl")
        type="test"
        self.test_trajectories = joblib.load(dir + self.env.name +  "_" + type + "_" + str(self.dataset_id) +"_transitions.pkl")
        # self.traj_set, self.train_rets = self.process_trajectories(self.trajectories)
        # self.test_traj_set, self.test_rets = self.process_trajectories(self.test_trajectories)
        type="eval"
        self.eval_trajectories = joblib.load(dir + self.env.name +  "_" + type  +"_transitions.pkl")

#
    def load_policies(self, dir="policies/"):
        self.policy_eval = joblib.load(dir + self.env.name + '_beh.pkl')

        self.policy_beh =  joblib.load( dir+ self.env.name + '_eval.pkl')

    def  get_deterministic(self, eval_probs):
        probs = torch.zeros(eval_probs.shape)
        n=eval_probs.shape[0]
        probs[np.arange(n),torch.argmax(eval_probs,dim=1)]=1.0
        return probs

    def construct_average_state_action_features(self,next_states):
        if isinstance(next_states,np.ndarray)==True:
            next_states = torch.tensor(next_states)

        n = next_states.shape[0]

        nsa_avg_features=[]
        eval_probs = self.policy_eval.predict_proba(next_states, transformed=self.config.transformed, normalized=self.config.normalized,scaler=self.scaler).detach()
        # eval_probs = self.get_deterministic(eval_probs)
        for action in range(self.env.action_dim):
            next_actions = (torch.ones(next_states.shape[0])*action).long()
            nsa_features = self.construct_state_action_features(next_states, next_actions)
            nsa_features =  nsa_features * eval_probs[np.arange(n), next_actions.flatten()].reshape(-1,1)
            nsa_avg_features.append(nsa_features)
        nsa_avg_features = torch.stack(nsa_avg_features)
        avg_features = torch.sum(nsa_avg_features,dim=0)
        return avg_features

    def process_trajectories(self, trajectories):
        trajectorySet = TrajectoriesDataSet()
        discounted_returns = []
        max_traj_length =0.0
        print("gamma",self.gamma)

        for trajectory in trajectories:
            processedTrajectory = ProcessedTrajectory(trajectory,self.gamma)
            discounted_returns.append(processedTrajectory.disc_rets)
            trajectorySet.add_trajectory(processedTrajectory)
            max_traj_length = np.maximum(max_traj_length,processedTrajectory.length)
        return trajectorySet, np.mean(discounted_returns), max_traj_length


    def extract_data(self):

        self.extract(self.trajectories)


        print("unique actions", np.unique(self.actions))

        self.init_dist = torch.tensor(self.init_dist)


        n = self.states.shape[0]
        if self.states.shape[0] > self.num_samples:
            self.states = self.states[:self.num_samples,:]
            self.actions = self.actions[:self.num_samples]
            self.next_states = self.next_states[:self.num_samples,:]
            self.rewards = self.rewards[:self.num_samples]
            self.steps = self.steps[:self.num_samples]


        self.state_dim = self.states.shape[1]


        self.actual_states = self.states.clone().detach()
        self.actual_next_states = self.next_states.clone().detach()

        self.state_action_features = self.construct_state_action_features(self.states, self.actions)
        self.next_state_action_features = self.construct_average_state_action_features(self.next_states)

        self.init_state_action_features = self.construct_average_state_action_features(self.init_states)




    def compute_pair_wise(self, num_pts=1000):
        states = self.states[:num_pts]
        distances=[]
        for idx in range(num_pts):
            for jdx in range(idx+1,num_pts,1):
                dist=0.0
                if self.type=="l1":
                    dist = torch.linalg.norm(states[idx]-states[jdx],ord=1)
                elif self.type=="l2":
                    dist = torch.linalg.norm(states[idx]-states[jdx],ord=2)
                elif self.type=="linf":
                    dist = torch.linalg.norm(states[idx]-states[jdx],ord=np.inf)

                else:
                    print("projection type invalid")
                    exit(1)
                distances.append(dist)
        avg_dist = np.mean(distances)
        return avg_dist





    def extract(self, trajectories):
        """
        Extracts states, actions, rewards, next_states, time_steps from transitions
        :param transitions: Array of n tuples of the form [s,a,r,s',time_step]
        :param policy_eval: reference to evaluation policy
        :return: states (n x state_dim), actions (n x 1) , next_states (n x state_dim) , rewards (n x 1), timesteps (n x 1)
        """
        actions = []
        rewards = []
        states = []
        next_states = []
        steps = []
        init_states = []
        init_dist=[]
        all_states=[]
        for transitions in trajectories:
            for idx in range(len(transitions)):
                if idx ==0:
                    state = transitions[idx].state.flatten()
                    state = torch.tensor(state)
                    # tr_state = self.policy_eval.forward1(state).detach().numpy().flatten()
                    init_states.append(state)
                init_dist.append(idx==0)
                actions.append(transitions[idx].action)
                state = transitions[idx].state.reshape(1,-1)
                next_state = transitions[idx].next_state.reshape(1,-1)
                state = torch.tensor(state)
                # tr_state = self.policy_eval.forward1(state).detach().numpy().flatten()
                next_state = torch.tensor(next_state)
                # tr_next_state = self.policy_eval.forward1(next_state).detach().numpy().flatten()

                states.append(state)
                next_states.append(next_state)

                rewards.append(transitions[idx].reward)
                steps.append(transitions[idx].done)

        actions = np.array(actions)
        rewards = np.array(rewards)
        steps = np.array(steps)
        states = torch.stack(states).reshape(-1,states[0].shape[-1])
        next_states = torch.stack(next_states).reshape(-1,next_states[0].shape[-1])
        init_states = torch.stack(init_states).reshape(-1,init_states[0].shape[-1])

        self.states,  self.next_states, self.init_states = torch.tensor(
            states),  torch.tensor(next_states),  torch.tensor(init_states)

        self.actions = torch.tensor(actions)
        self.rewards = torch.tensor(rewards)
        self.steps = torch.tensor(steps)
        self.init_dist = torch.tensor(init_dist)


        if self.config.transformed == True:
            self.states = self.policy_eval.forward1(self.states).detach()
            self.next_states = self.policy_eval.forward1(self.next_states.double()).detach()
            self.init_states = self.policy_eval.forward1(self.init_states.double()).detach()

        if self.config.normalized == True:
             self.scaler = StandardScalerTensor()
             self.states = self.scaler.fit_transform(self.states)
             self.next_states = self.scaler.transform(self.next_states)
             self.init_states = self.scaler.transform(self.init_states)

        # if config.random_features == True:
        #     states = self.states.detach().cpu().numpy()
        #     next_states = self.next_states
        #     init_states = self.init_states
        #     self.scaler = StandardScalerTensor()
        #     states = self.scaler.fit_transform(states)
        #     next_states = self.scaler.transform(next_states)
        #     init_states = self.scaler.transform(init_states)
        #     transformer = GaussianRBFTensor(n_centers=self.config.n_centers,low=np.ones(states.shape[1])*-1,high=np.ones(states.shape[1]))
        #     states = transformer.transform(states)
        #     next_states = transformer.transform(next_states)
        #

        self.states = self.states.detach()
        self.next_states = self.next_states.detach()
        self.init_states = self.init_states.detach()


        # if self.type=="l1":
        #     self.eps = self.frac * torch.std(torch.linalg.norm(self.states,dim=1,ord=1)).detach().cpu().numpy()
        # elif self.type=="l2":
        #     self.eps = self.frac * torch.std(torch.linalg.norm(self.states,dim=1,ord=2)).detach().cpu().numpy()
        # else:
        #     self.eps = self.frac * torch.std(torch.linalg.norm(self.states,dim=1,ord=torch.inf)).detach().cpu().numpy()
        #

        self.eps = self.frac * self.compute_pair_wise()



    def reset_eps(self):

        self.eps = self.frac * self.compute_pair_wise()


    def get_init_state(self):
        """
        Returns initial state
        :return:
        """

        return self.init_states

    def set_next_state(self, next_states):
        """
        Sets self.next_states to given next_states
        """
        next_states = np.array(next_states)
        next_states = torch.tensor(next_states)
        self.next_states = next_states

        self.next_state_action_features = self.construct_average_state_action_features(self.next_states)


    def set_state(self, states):
        """
        Sets self.next_states to given next_states
        """
        # set states as cu
        states = np.array(states)
        states = torch.tensor(states)
        self.btr_states = states.clone().detach()
        self.states = states
        self.state_action_features = self.construct_state_action_features(self.states, self.actions)


    def get_next_state_features(self,requires_grad=True):
        """
        Sets self.next_states to given next_states
        """
        next_states = self.next_states.clone()

        next_states = next_states.requires_grad_(requires_grad)
        return next_states



    def get_state_features(self,requires_grad=True):
        """
        Sets self.next_states to given next_states
        """

        states = self.states.clone()

        states = states.requires_grad_(requires_grad)
        # returns states and transformed features
        return states


    def get_state_features_indices(self,requires_grad=True, indices=None):
        """
        Sets self.next_states to given next_states
        """

        inverted_state = self.states.clone().detach()[indices.flatten(),:]
        states = inverted_state.requires_grad_(requires_grad)
        from datetime import datetime

        return states


    def get_state_action_features_indices(self,requires_grad=True, indices=None):
        """
        Sets self.next_states to given next_states
        """

        inverted_state = self.state_action_features.clone().detach()[indices.flatten(),:]
        states = inverted_state.requires_grad_(requires_grad)
        from datetime import datetime

        return states


    def construct_state_action_features(self, states, actions):
            """
            Constructs state-action features from state and actions arrays
            :param states:
            :param actions:
            :param num_actions:
            :return:
            """

            num_actions = self.env.action_dim
            if isinstance(actions,np.ndarray):
                actions = torch.tensor(actions).long()

            state_action_features = []
            actions = np.array(actions).flatten()
            actions = torch.tensor(actions).long()
            for idx in range(states.shape[0]):
                # Array of state-action features
                state_temp = torch.zeros((states[idx].shape[0]) * num_actions)
                # Sets state_temp[a *state_dim: (a+1)*state_dim] = state_features
                state_temp[
                int(actions[idx] * (states[idx].shape[0])):int((actions[idx] + 1) * (states[idx].shape[0]))] = \
                states[
                    idx]
                state_action_features.append(state_temp)
            return torch.stack(state_action_features)

            # return states


    def reset(self):
        self.states = self.actual_states.clone().detach()

        self.next_states = self.actual_next_states.clone().detach()

        self.state_action_features = self.construct_state_action_features(self.states, self.actions)

        self.next_state_action_features = self.construct_average_state_action_features(self.next_states)
        self.init_state_action_features = self.construct_average_state_action_features(self.init_states)



class StandardScalerTensor():
    def __init__(self):
        self.mean =None
        self.std = None

    def fit_transform(self,X):
        if self.mean is None:
            self.mean = torch.mean(X,dim=0).reshape(1,-1)
            self.std = torch.std(X,dim=0).reshape(1,-1)
        X_normalized = X-self.mean.reshape(1,-1)/self.std.reshape(1,-1)
        return X_normalized


    def transform(self,X):
        if self.mean is None:

            print("Model not fit yet. Call fit_transform!")
            assert(0)
        X_normalized = (X-self.mean)/(self.std + 1e-5)
        return X_normalized

    def inverse_transform(self,X):
        if self.mean is None:
             print("Model not fit yet. Call fit_transform!")
             assert(0)
        X_scaled = (X*self.std)+self.mean

        return X_scaled





