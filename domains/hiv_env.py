import sys
import pickle
sys.path.append("domains/HIV/hiv_domain/")
sys.path.append("domains/HIV/hiv_domain/hiv_simulator/")

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
from hiv import HIVTreatment as model

import os
with open('domains/HIV/hiv_domain/hiv_simulator/hiv_preset_hidden_params', 'rb') as f:
    preset_hidden_params = pickle.load(f, encoding='latin1')

class HIVEnv():
    def __init__(self,discount_factor=0.98):
        self.task = model(perturb_rate=0.03)

        self.action_dim = 4
        self.preset_params = preset_hidden_params[20]
        self.state_dim = 6
        self.gamma = discount_factor
        self.discount_factor=discount_factor
        self.reset()
        self.name="hiv_domain"
        self.episode_length=50


    def reset(self):
        self.task.reset(perturb_params=True, **self.preset_params)
        state = self.task.observe()
        return state.flatten()


    def step(self,action):
        rewards, obs = self.task.perform_action(action, perturb_params=True, **self.preset_params)
        done = self.task.is_done(episode_length=self.episode_length)

        return obs, rewards, done


