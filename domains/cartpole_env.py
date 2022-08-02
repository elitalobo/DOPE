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
#import gym_sepsis
import gym

import pickle
RANDOM_SEED=10
class CartpoleEnv():
    def __init__(self, discount_factor=0.98):
        self.env = gym.make("CartPole-v0")
        self.env.action_space.seed(args.seed)

        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.gamma = discount_factor
        self.discount_factor=discount_factor
        self.reset()
        self.name="cartpole_domain"


    def reset(self):
        obs = self.env.reset().flatten()
        return obs


    def step(self,action):
        obs, rewards, dones, info = self.env.step(action)
        return obs.flatten(), rewards, dones


