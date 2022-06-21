
import sys
import os
import sys
import numpy as np
seed = np.random.randint(1,10)
sys.path.append("experiments/")
sys.path.append("src/")
from create_datasets  import *
# seed=10
import torch

torch.manual_seed(seed)
import random

random.seed(seed)

np.random.seed(seed)


sys.path.append("domains/HIV/src/")
sys.path.append("domains/HIV/hiv_domain/")
sys.path.append("domains/HIV/")
sys.path.append("domains/")
sys.path.append("domains/Cancer/")
sys.path.append("domains/rlsepsis234/")
#from sepsis_env import *
sys.path.append("domains/HIV/")
sys.path.append("policies/")
from main_hiv import *
from time import time
import joblib
from collections import deque
from memory import *
from utils import *
from models import MDPnet
from config import hiv_config
from fitted_Q_learning import FittedQIteration as Qlearning
from deep_fitted_Q_learning import DeepFittedQIteration as DeepQlearning
from policies import *
from config import *
from train_pipeline import mdpmodel_train, mdpmodel_test
from hiv_env import  *
from cancer_env import *
from cartpole_env import *
from mountaincar_env import *
from custom import *
from src import *
from models import QNet
from train_pipeline import train_pipeline
from utils_nn import load_qnet, error_info

def preprocess(state,config):
    new_state = (state-config["state_mean"])/(config["state_std"])
    return new_state

def preprocess_r(reward,config):
    #print(config["reward_std"])
    if config["normalize_rewards"]==True:
        new_reward = (reward-config["reward_mean"])/(config["reward_std"])
    else:
        new_reward = reward
    return new_reward


datasets=[]
for idx in range(25,30,1):
    config = hiv_config
    env = HIVEnv(discount_factor=config.gamma)
    generate_datasets(env,config,behavior_eps=0.5,id=idx,select_max=False)
    print("num_trajectories",config.num_trajectories)
# env = CustomEnv()
# config = custom_config
# print("num_trajectories",config.num_trajectories)
# generate_eval_datasets(env,config,select_max=True)
# print(traj)
