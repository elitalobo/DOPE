
import sys
import os
import sys
import numpy as np
seed = np.random.randint(1,10)
sys.path.append("experiments/")
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


def extract_datasets(traj_set,config_g):
    config=dict()
    states=[]
    rewards = []
    for trajectory in traj_set:
        transitions = []
        idx = 0
        for transition in trajectory:
            state = transition.state.flatten()
            states.append(state)
            rewards.append(transition.reward)
    config["reward_mean"]=np.mean(rewards)
    config["reward_std"]=np.std(rewards)
    config["state_mean"]=np.mean(states,axis=0)
    config["state_std"]=np.mean(states,axis=0)
    config["normalize_rewards"] = config_g.normalize_rewards

    trajectories = []
    states=[]
    actions=[]
    next_states=[]
    rewards=[]
    for trajectory in traj_set:
        transitions = []
        idx = 0
        lent=len(trajectory)
        for transition in trajectory:
            state = transition.state.flatten() #preprocess(transition.state.flatten(),config)
            action = transition.action
            reward = preprocess_r(transition.reward,config)
            next_state = transition.next_state.flatten() #preprocess(transition.next_state.flatten(),config)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)

            transitions.append(Transition(
               state,action,next_state,reward,idx==lent,None))

            idx += 1
        trajectories.append(transitions)

    # np.save("datasets/hiv_states.npy",np.array(states))
    # np.save("datasets/hiv_actions.npy", np.array(actions))
    # np.save("datasets/hiv_next_states.npy",np.array(next_states))
    # np.save("datasets/rewards.npy", np.array(rewards))



    return trajectories, config

def generate_eval_datasets(env,config,select_max,behavior_eps=0.0):
    if env.name in ["cartpole_domain","mountaincar_domain","cancer_domain","hiv_domain","custom"]:
        qiter = DeepQlearning(env=env, ins=20, config=config)


        eval_qnet = QNet(config.state_dim, config.dqn_hidden_dims, config.action_size)

        load_qnet(eval_qnet, checkpoint="policies/", filename=env.name + '_qnet.pth.tar')
        eval_qnet.eval()
        qiter.qnet = eval_qnet
        traj_set, scores = qiter.generate_trajectories(config.num_trajectories*2, behavior_eps=behavior_eps,select_max=select_max)


        transitions_train, config_new = extract_datasets(traj_set, config)
        if os.path.exists('datasets') == False:
            os.makedirs('datasets')
        name="eval"
        joblib.dump(transitions_train, 'datasets/' + env.name + "_" + name +   "_transitions.pkl")



def generate_datasets(env,config,gamma=0.98,args=None,behavior_eps=0.05, name="test", id=1, select_max=False):

    if env.name in ["cartpole_domain","mountaincar_domain","cancer_domain","hiv_domain","custom"]:
        qiter = DeepQlearning(env=env, ins=20, config=config)


        eval_qnet = QNet(config.state_dim, config.dqn_hidden_dims, config.action_size)

        load_qnet(eval_qnet, checkpoint="policies/", filename=env.name + '_qnet.pth.tar')
        eval_qnet.eval()
        qiter.qnet = eval_qnet
        traj_set, scores = qiter.generate_trajectories(config.num_trajectories, behavior_eps=behavior_eps,select_max=select_max)

        traj_set_test, scores_test = qiter.generate_trajectories(config.num_trajectories, behavior_eps=behavior_eps,select_max=select_max)

        transitions_train, config_new = extract_datasets(traj_set, config)
        transitions_test, config_new = extract_datasets(traj_set_test, config)
        if os.path.exists('datasets') == False:
            os.makedirs('datasets')
        name="train"
        joblib.dump(transitions_train, 'datasets/' + env.name + "_" + name + "_" + str(id) +   "_transitions.pkl")
        name="test"
        joblib.dump(transitions_test, 'datasets/' + env.name + "_" + name + "_" + str(id) + "_transitions.pkl")

        if id==1:
            joblib.dump(config, "datasets/" + env.name  +  "_config.pkl")

            policy_eval = NNPolicy(qiter.qnet,config.state_dim, config.action_size)
            policy_beh = NNPolicy(qiter.qnet, config.state_dim, config.action_size,eps_behavior=0.05)

            joblib.dump(policy_eval,"policies/"+ env.name + "_eval.pkl")
            joblib.dump(policy_beh,"policies/"+ env.name + "_beh.pkl")

        # return config_new, transitions_train, transitions_test, policy_eval, policy_beh

    else:

        env_fq = Qlearning(env,episode_length=config.max_length)
        env_fq.tree = joblib.load('policies/' + env.name + "_"+ 'extra_tree_gamma_ins20.pkl')

        eval_env = Qlearning(env,
                               episode_length=config.max_length)
        eval_env.tree= joblib.load('policies/' + env.name +"_"+ 'extra_tree_gamma_ins20.pkl')


        traj_set, scores = env_fq.generate_trajectories(config.num_trajectories, behavior_eps=behavior_eps)

        traj_set_test, scores_test = env_fq.generate_trajectories(config.num_trajectories,behavior_eps=behavior_eps)

        transitions_train, config_new = extract_datasets(traj_set,config)
        transitions_test, config_new = extract_datasets(traj_set_test,config)
        if os.path.exists('datasets')==False:
            os.makedirs('datasets')

        joblib.dump(transitions_train,'datasets/' + env.name + "_transitions.pkl")
        joblib.dump(transitions_test,'datasets/' + env.name + "_transitions.pkl")


        joblib.dump(config,"datasets/" + env.name + "_config.pkl")



        policy_eval = TreePolicy(env_fq)
        hiv_policy_beh = TreePolicy(eval_env,eps_behavior=0.05)

        # return config_new, transitions_train, transitions_test, policy_eval, hiv_policy_beh

# datasets=[]
# for idx in range(0,10,1):
#     config = cancer_config
#     env = CancerEnv(discount_factor=config.gamma)
#     generate_datasets(env,config,behavior_eps=0.05,id=idx,select_max=True)
#     print("num_trajectories",config.num_trajectories)
#
# env = HIVEnv()
# config = hiv_config
# # print("num_trajectories",config.num_trajectories)
# generate_eval_datasets(env,config,select_max=True)
# print(traj)