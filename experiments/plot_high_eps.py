# import Cancer

import os

from datetime import datetime
import sys
sys.path.append("plots1/")
sys.path.append("domains/")
sys.path.append("algorithms_ope/")
from argparser_fqe import parse
from cancer_env import *
from hiv_env import *
from sepsis_env import *
from cartpole_env import *
from custom import *
from mountaincar_env import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
args = parse()
import torch
from matplotlib import collections  as mc
from sklearn.utils import resample

torch.manual_seed(args.seed)
import random
import scipy
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
print(sys.path)
# sys.path.append("/Users/elitalobo/PycharmProjects/untitled/soft-robust/interpretable_ope_public/")
sys.path.append("../domains_for_bilevel/")
# import Cancer


sys.path.append("algorithms_ope/")
from fqe_method import *
from importance_sampling_method import *
from wdr_method import *
from utils import *
from scipy.stats import bootstrap
import seaborn as sns

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
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def boostrap_intervals(data,confidence_level=0.95,axis=0, iqm=True):
    method = np.median
    if len(np.unique(data))==1:
        return data.flatten()[0], data.flatten()[0]
    # print("here")
    if iqm== True:
        method = interquartile_mean
        # print("got iqm")
    rng = np.random.default_rng()

    try:

        res = bootstrap((data,), method, confidence_level=confidence_level,
                        random_state=rng,method='percentile', n_resamples=10000)
        # print("success")
    except:
        print("error here")
        data = np.sort(data)
        return data.flatten()[0], data.flatten()[-1]
    return res.confidence_interval.low,res.confidence_interval.high

def interquartile_mean(data,axis=None):
    if axis==None:
        axis=-1
    mean = scipy.stats.trim_mean(data, proportiontocut=0.25, axis=axis)
    return mean


def get_ci_intervals(data):
    data = np.array(data)
    if len(data.shape)==2:
        data = data.reshape((1,data.shape[0],data.shape[1]))
    cis_lower=[]
    cis_upper = []
    for idx in range(data.shape[1]):
        lower_arr=[]
        upper_arr=[]
        for jdx in range(data.shape[2]):
            lower, upper = boostrap_intervals(data[:,idx,jdx])
            lower_arr.append(lower)
            upper_arr.append(upper)
        cis_lower.append(lower_arr)
        cis_upper.append(upper_arr)
    cis_lower = np.array(cis_lower)
    cis_upper = np.array(cis_upper)
    return cis_lower, cis_upper





class Plotter():
    def __init__(self, path, dir="plots/",axis=0):
        self.path = path
        self.axis=axis
        self.budget_range= np.arange(0.0,0.51,0.05)
        self.eps_range =  np.arange(0.02,0.12,0.02)
        self.discount_range = np.array([0.2,0.4,0.6,0.8,0.98])
        self.projections=["l1","l2","l∞"]
        self.plot_dir= dir
        self.colors = ['#dc267f', '#648fff', '#fe6100', '#785ef0', '#ffb000','#7f7f7f', '#bcbd22', '#17becf']
        # self.colors=['r','g','b']
        self.c=["mistyrose","lightgreen","skyblue"]
        self.signs={"hiv":-1, "custom":1, "mountaincar":1, "cartpole":1,"cancer":1}
        if os.path.exists("plots1/experiments1") == False:
            # os.rmdir("plots1/experiments1")
            os.mkdir("plots1/experiments1")

        if os.path.exists("plots1/experiments2")  == False:
            # os.rmdir("plots1/experiments2")
            os.mkdir("plots1/experiments2")

        if os.path.exists("plots1/experiments3")  == False:
            # os.rmdir("plots1/experiments3")
            os.mkdir("plots1/experiments3")

        if os.path.exists("plots1/experiments4")  == False:
            # os.rmdir("plots1/experiments4")
            os.mkdir("plots1/experiments4")

        if os.path.exists("plots1/experiments5") == False:
            # os.rmdir("plots1/experiments5")
            os.mkdir("plots1/experiments5")

        if os.path.exists("plots1/experiments6") == False:
            # os.rmdir("plots1/experiments5")
            os.mkdir("plots1/experiments6")

        if os.path.exists("plots1/experiments7") == False:
            # os.rmdir("plots1/experiments5")
            os.mkdir("plots1/experiments7")

        if os.path.exists("plots1/experiments8") == False:
            # os.rmdir("plots1/experiments5")
            os.mkdir("plots1/experiments8")

        if os.path.exists("plots1/experiments9") == False:
            # os.rmdir("plots1/experiments5")
            os.mkdir("plots1/experiments9")


        self.func =  interquartile_mean

    def plot(self, file_name,res,labels,x_vals,xlabel,ylabel,std_l,std_u, add=False):
        plt.clf()
        res = np.round(res,2)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12.0, 10.0)
        plt.rcParams.update({'font.size': 25})
        plt.rc('legend',fontsize=28)
        plt.xticks(fontsize=25, rotation=0)
        plt.yticks(fontsize=25)
        if add==True:
            new_res = np.zeros((res.shape[0],res.shape[1]+1))
            new_std_l = np.zeros((res.shape[0],res.shape[1]+1))
            new_std_u = np.zeros((res.shape[0],res.shape[1]+1))
            new_x = x_vals.tolist()
            new_x.insert(0,0.0)
            new_res[:,1:]=res
            new_std_l[:,1:]=std_l
            new_std_u[:,1:]=std_u
            res = np.array(new_res)
            std_l = np.array(new_std_l)
            std_u = np.array(new_std_u)
            x_vals = np.array(new_x)
        for idx in range(res.shape[0]):

            plt.plot(x_vals,res[idx,:], label=labels[idx],c=self.colors[idx],marker='o',linewidth=7)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            lower = std_l[idx,:]
            upper = std_u[idx,:]
            indices = lower  < 0
            lower[indices]=0.0
            plt.fill_between(x_vals, lower, upper,
                             alpha=0.3, facecolor=self.colors[idx])
            plt.ylim(np.min(std_l),np.max(std_u))
            plt.xlim(np.min(x_vals),np.max(x_vals))


        plt.savefig(self.plot_dir + file_name+ "_" + str(self.axis))
        # plt.clf()

    def experiment1(self):
        res={}
        for file in os.listdir(self.path +"/"):
            try:
                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment1" not in file:
                    continue
                dataset_id = int(name[3])
                if dataset_id not in np.arange(25,30):
                    continue
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[1] + "_" + name[2]
                else:
                    method = name[1]
                env = name[5]
                attack_type = name[4]
                key = method + "-" + env + "-" + attack_type
                if "_policy1" in file:
                    plt.clf()
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[2:]
                    if returns.shape[0] < 3*len(self.budget_range):
                        continue
                    rets = returns.reshape(3,-1)
                    if res.get(key) is None:
                        res[key] = []
                    rets = np.abs(100*(rets - init_ret))/(np.abs(init_ret)+ 1e-10)

                    res[key].append(rets)

            except:
                print(file)
        for key, value in res.items():
            try:
                value1 = self.func(value,0)[:2,:]

                if key =="IS_pdis-hiv-influence":
                    print("here key")
                std_lower, std_upper = get_ci_intervals(value)
                std_lower = std_lower[:2,:]
                std_upper = std_upper[:2,:]

                xlabel="Attacker's budget (frac)"
                ylabel="Percentage error in value function"
                self.plot("experiments1/" + key + "_experiment1_higheps",value1,self.projections[:2],self.budget_range,xlabel,ylabel,std_lower,std_upper)
            except:
                print("failedhere")
                print(key)
                pass

        return res


        # pass
    def experiment2(self):
        res={}
        for file in os.listdir(self.path +"/"):
            try:
                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment2" not in file:
                    continue
                dataset_id = name[3]

                method = name[1]
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[1] + "_" + name[2]
                env = name[5]
                attack_type = name[4]
                key = method + "-" + env + "-" + attack_type
                if "_policy1" in file:
                    plt.clf()
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[2:]
                    if returns.shape[0] < 3*len(self.eps_range):
                        continue
                    rets = returns.reshape(3,-1)
                    if res.get(key) is None:
                        res[key] = []
                    rets = np.abs(100*(rets - init_ret))/(np.abs(init_ret) +  1e-10)
                    if init_ret==0.0:
                        print(file)

                    res[key].append(rets)

            except:
                print(file)
        for key, value in res.items():
            try:
                value1 = self.func(value,0)[:2,:]
                value = np.array(value)
                std_lower, std_upper = get_ci_intervals(value)
                std_lower = std_lower[:2,:]
                std_upper = std_upper[:2,:]

                xlabel="Percentage of corrupt points (α)"
                ylabel="Percentage error in value function"
                self.plot("experiments2/" + key + "experiment2",value1,self.projections[:2],self.eps_range,xlabel,ylabel,std_lower,std_upper,add=True)
            except:
                print(key)


        return res

    def experiment3(self):
        res={}
        res_train={}
        for file in os.listdir(self.path +"/"):
            try:
                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment3" not in file:
                    continue
                dataset_id = name[3]
                method = name[3]
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[3] + "_" + name[4]
                env = name[7]
                attack_type = name[6]
                key = method + "-" + env + "-" + attack_type
                if "_policy1" and "before_attack" in file:
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[0:]
                    returns = returns.reshape((-1,2))[:,0]
                    rets = returns.reshape(3,-1)
                    if res.get(key) is None:
                        res[key] = []
                    # rets = (100*(rets - init_ret))/(init_ret)

                    res[key].append(rets)
                if "_policy1" and "after_attack" in file:
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[2:]
                    # returns = returns.reshape((-1,2))[:,0]

                    rets = returns.reshape(3,-1)
                    if res_train.get(key) is None:
                        res_train[key] = []

                    # rets = (100*(rets - init_ret))/(init_ret)
                    res_train[key].append(rets)
            except:
                print(file)
        for key, value in res.items():
            try:
                value1 = value
                value2 = res_train[key]
                value = np.abs(np.array(value2)-np.array(value1))/np.abs(np.array(value1))
                value1 = self.func(value,0)[:2,:]
                std_lower, std_upper = get_ci_intervals(value)
                std_lower = std_lower[:2,:]
                std_upper = std_upper[:2,:]

                xlabel="Discount factor"
                ylabel="Percentage error in value function"
                self.plot("experiments3/" + key + "experiment3",value1,self.projections[:2],self.discount_range,xlabel,ylabel,std_lower, std_upper,add=False)

            except:
                pass
        # for key, value in res_train.items():
        #     value = self.func(value,0)
        #     xlabel="discount factor"
        #     ylabel="Percentage error in value function"
        #     self.plot(key + '_experiment3',value,self.projections,self.discount_range,xlabel,ylabel)
        return res
    def experiment4(self, res_new):
        res={}
        res_train={}
        for file in os.listdir(self.path +"/"):
            try:

                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment4" not in file:
                    continue
                if "experiment5" in file:
                    continue
                dataset_id = name[2]
                method = name[1]
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[1] + "_" + name[2]
                env = name[5]
                attack_type = name[4]
                key = method + "-" + env + "-" + attack_type
                key_alt = method + "-" + env + "-" + "influence"
                if "_policy1" in file:
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[2:]
                    lent = len(self.budget_range)*50*(self.axis+1)
                    l = len(self.budget_range)*50
                    if len(returns)< lent:
                        continue
                    else:
                        returns = returns[self.axis*l:l*(self.axis+1)]
                    if "cartpole" in key and "FQE" in key:
                        print("found")

                    # num = np.int(np.floor(lent/8))
                    # lent = num*self.budget_range.shape[0]
                    # returns = returns[:lent]
                    rets = returns.reshape((1,self.budget_range.shape[0],-1))
                    rets = np.mean(rets,axis=2)
                    rets = np.abs(100*(rets - init_ret))/(np.abs(init_ret)+ 1e-10)
                    if res.get(key_alt) is None:
                        res[key_alt] = []
                    res[key_alt].append(rets)
                # if "_pbe" in file:
                #     plt.clf()
                #     returns = np.load(full_path)
                #     rets = returns.reshape(3,-1)
                #     rets = (100*(rets - init_ret))/(init_ret)
                #     if res.get(key) is None:
                #         res[key] = []
                #     res_train[key].append(rets)
            except:
                print(file)

        for key, value in res.items():
            try:
                if "cartpole" in key and "FQE" in key:
                    print("found")

                value1 = self.func(value,0).reshape(1,-1)
                std1_lower, std1_upper = get_ci_intervals(value)
                std1_lower = std1_lower[0,:]
                std1_upper = std1_upper[0,:]

                xlabel="Attacker's budget (frac)"
                ylabel="Percentage error in value function"

                bs = res_new[key]
                value2 = self.func(bs,0)[self.axis,:].reshape(1,-1)
                std2_lower, std2_upper = get_ci_intervals(bs)
                std2_lower = std2_lower[self.axis,:]
                std2_upper = std2_upper[self.axis,:]

                final_val = np.stack((value1,value2),axis=0).squeeze()
                final_std_lower = np.stack((std1_lower,std2_lower),axis=0).squeeze()
                final_std_upper = np.stack((std1_upper,std2_upper),axis=0).squeeze()

                self.plot("experiments4/" + key + "experiment4",final_val,["Random Attack","DOPE Attack"],self.budget_range,xlabel,ylabel,final_std_lower,final_std_upper,add=False)
            except:
                pass
        # for key, value in res_train.items():
        #     value = self.func(value,0)
        #     xlabel="attacker's budget"
        #     ylabel="Percentage error in value function (Random Attack)"
        #     self.plot(key + '_experiment4',value,self.projections,self.budget_range,xlabel,ylabel)
        return res

    def experiment5(self,res_new, res_random):
        res={}
        res_train={}
        for file in os.listdir(self.path +"/"):
            try:
                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment6" not in file:
                    continue
                dataset_id = name[3]
                method = name[1]
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[1] + "_" + name[2]
                env = name[5]
                attack_type = name[4]
                key = method + "-" + env + "-" + attack_type
                key_alt = method + "-" + env + "-" + "influence"
                if "_policy1" in file:
                    plt.clf()
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[2:]
                    lent = len(self.budget_range)*50
                    l = len(self.budget_range)*50
                    if len(returns)< lent:
                        continue
                    else:
                        returns = returns[l*self.axis:(self.axis+1)*lent]
                    rets = returns.reshape((1,self.budget_range.shape[0],-1))
                    rets = np.mean(rets,axis=2)
                    if res.get(key_alt) is None:
                        res[key_alt] = []
                    rets = np.abs(100*(rets - init_ret))/(np.abs(init_ret)+ 1e-10)
                    res[key_alt].append(rets)

            except:
                print(file)
        # for key, value in res.items():
        #     try:
        #         value1 = self.func(value,0)
        #         std = np.std(value,0)
        #         xlabel="Attacker's budget"
        #         ylabel="Percentage error in value function"
        #         self.plot("experiments5/" + key + "experiment5",value1,self.projections,self.budget_range,xlabel,ylabel,std,add=False)
        res_all = {}
        for key, value in res.items():
            method =key.split("-")[0]
            env =key.split("-")[1]

            try:
                value1 = self.func(value,0).reshape(1,-1)
                # std1 = np.std(value,0)[0,:].reshape(1,-1)
                std1_lower, std1_upper = get_ci_intervals(value)
                std1_lower = std1_lower[0,:]
                std1_upper = std1_upper[0,:]


                bs = res_new[key]
                value2 = self.func(bs,0)[self.axis,:].reshape(1,-1)
                # std2 = np.std(bs,0)[0,:].reshape(1,-1)
                std2_lower, std2_upper = get_ci_intervals(bs)
                std2_lower = std2_lower[self.axis,:]
                std2_upper = std2_upper[self.axis,:]


                rs = res_random[key]
                value3 = self.func(rs,0)[0,:].reshape(1,-1)
                # std3 = np.std(rs,0)[0,:].reshape(1,-1)
                std3_lower, std3_upper = get_ci_intervals(rs)
                std3_lower = std3_lower[0,:]
                std3_upper = std3_upper[0,:]

                final_val = np.stack((value2,value1, value3),axis=0).squeeze()
                final_std_lower = np.stack((std2_lower,std1_lower, std3_lower),axis=0).squeeze()
                final_std_upper = np.stack((std2_upper,std1_upper, std3_upper),axis=0).squeeze()




                if res_all.get(env) is None:
                    res_all[env]={}
                    res_all[env]["DOPE Attack"]={}
                    res_all[env]["Random DOPE Attack"]={}
                    res_all[env]["Random Attack"]={}
                res_all[env]["DOPE Attack"]["mean"]=value2
                res_all[env]["DOPE Attack"]["std_lower"]=std2_lower
                res_all[env]["DOPE Attack"]["std_upper"]=std2_upper

                res_all[env]["Random DOPE Attack"]["mean"]=value1
                res_all[env]["Random DOPE Attack"]["std_lower"]=std1_lower
                res_all[env]["Random DOPE Attack"]["std_upper"]=std1_upper


                res_all[env]["Random Attack"]["mean"]=value3
                res_all[env]["Random Attack"]["std_lower"]=std3_lower
                res_all[env]["Random Attack"]["std_upper"]=std3_upper


                xlabel="Attacker's budget (frac)"
                ylabel="Percentage error in value function"
            except:
                continue


            self.plot("experiments5/" + key + "experiment5",final_val,["DOPE","Random DOPE","Random Attack"],self.budget_range,xlabel,ylabel,final_std_lower, final_std_upper,add=False)
            # except:
            #     pass
            # except:
            #     pass
        return res_all

    def experiment7(self, res_new):
        res={}
        res_train={}
        for file in os.listdir(self.path):
            try:
                if "deletedreturns" not in file:
                    continue


                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment2" not in file:
                    continue
                dataset_id = name[3]

                method = name[1]
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[1] + "_" + name[2]
                env = name[5]
                attack_type = name[4]
                key = method + "-" + env + "-" + attack_type
                returns = np.load(full_path)
                init_ret = returns[0]
                tru_rets = returns[1]
                returns = returns[2:]
                rets = returns.reshape(3,-1)
                if res.get(key) is None:
                    res[key] = []
                rets = np.abs(100*(rets - init_ret))/np.abs(init_ret)

                res[key].append(rets)
            except:
                print(file)

        for key, value in res.items():
            try:

                value1 = self.func(value,0)[0,:].reshape(1,-1)
                # std1 = np.std(value,0)[0,:].reshape(1,-1)
                std1_lower, std1_upper = get_ci_intervals(value)
                std1_lower = std1_lower[0,:]
                std1_upper = std1_upper[0,:]


                bs = res_new[key]
                value2 = self.func(bs,0)[0,:].reshape(1,-1)
                # std2 = np.std(bs,0)[0,:].reshape(1,-1)
                std3_lower, std3_upper = get_ci_intervals(bs)
                std3_lower = std3_lower[0,:]
                std3_upper = std3_upper[0,:]


                final_val = np.concatenate((value2,value1),axis=0)
                final_std_lower = np.stack((std1_lower,std3_lower),axis=0)
                final_std_upper = np.stack((std1_upper,std3_upper),axis=0)


                xlabel="Percentage of corrupt points (α)"
                ylabel="Percentage error in value function"
                self.plot("experiments7/" + key + "_deleted",final_val,["DOPE","Deletion"],self.eps_range,xlabel,ylabel,final_std_lower,final_std_upper,add=True)
            except:
                print(key)
        return res
    def extract_states(self, trajectories):
        states = []
        next_states = []
        for transitions in trajectories:
            for idx in range(len(transitions)):
                state = transitions[idx].state.flatten()
                states.append(state)
                next_states.append(transitions[idx].next_state.flatten())
        states = np.array(states)
        next_states = np.array(next_states
                               )
        return states, next_states


    def plot_transitions(self,path):
        type="train"
        trajectories = joblib.load("datasets/" + "custom" +  "_" + type + "_"+ str(1) + "_transitions.pkl")
        #
        states,next_states = self.extract_states(trajectories)
        # corrupt_states = np.load(path + "/" + "corrupt_states.npy")
        # indices = np.load(path + "/" + "influential_indices.npy").tolist()
        lines1=[]
        lines2=[]
        color1=[]
        color2=[]
        import pylab as pl
        indices=[]

        fig, ax = pl.subplots(nrows=1,ncols=2)
        # corrupt_states= corrupt_states[:500,:]
        for idx in range(states.shape[0]):
            if idx in indices:
                color1.append('red')
                color2.append('green')
            else:
                color1.append('black')
                color2.append('black')
            lines1.append([(states[idx][0],states[idx][1]),(next_states[idx][0],next_states[idx][1])])
            lines2.append([(states[idx][0],states[idx][1]),(next_states[idx][0],next_states[idx][1])])

        print(lines1)
        print(lines2)
        from matplotlib import collections  as mc

        lc1 = mc.LineCollection(lines1, colors=color1, linewidths=2)
        lc2 = mc.LineCollection(lines2, colors=color2, linewidths=2)

        ax[0].add_collection(lc1)
        ax[1].add_collection(lc2)
        ax[1].autoscale()
        ax[0].autoscale()

        # ax[0].margins(0.1)
        # ax[1].margins(0.1)

        # import numpy as np
        import pylab as pl
        pl.clf()

        lines=[]
        c=[]
        # lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
        for idx in range(states.shape[0]):
            # lines.append([(states[idx][0],states[idx][1]),(next_states[idx][0],next_states[idx][1])])
            # if idx in indices:
            #     c.append('red')
            #     # c.append('green')
            # else:
            #     c.append('black')
            if idx in indices:
                color1.append('red')
                color2.append('green')
            else:
                color1.append('black')
                color2.append('black')
            lines1.append([(states[idx][0],states[idx][1]),(next_states[idx][0],next_states[idx][1])])
            lines2.append([(states[idx][0],states[idx][1]),(next_states[idx][0],next_states[idx][1])])


        lc = mc.LineCollection(lines1, colors=color1, linewidths=2)
        lc1 = mc.LineCollection(lines2, colors=color2, linewidths=2)

        fig, ax = pl.subplots(nrows=1,ncols=2)
        ax[0].add_collection(lc)
        ax[0].autoscale()
        ax[1].add_collection(lc1)
        ax[1].add_collection(lc1)
        ax[1].autoscale()
        # ax.margins(0.1)

        fig.savefig("plots1/custom_example.pdf")

    def plot_bar(self,key,value1, value2, std1, std2,ylabel,xticks):
        plt.clf()
        vals = np.array([value1, value2])
        err = np.array([std1, std2])
        x_pos=np.arange(len(xticks))
        plt.bar(x_pos, vals, yerr=err, align='center', alpha=0.5, ecolor='black', capsize=10)
        plt.ylabel(ylabel)
        plt.xticks(x_pos,xticks)
        # plt.title('Coefficent of Thermal Expansion (CTE) of Three Metals')
        plt.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(self.plot_dir + key + ".pdf")

    def get_sum_indices(self,weights, indices):
        weights = weights.flatten()
        indices = indices.flatten()
        return np.sum(np.abs(weights[indices]))

    def experiment6(self,flag=False):
        res={}
        res_indices={}
        for file in os.listdir(self.path +"/"):
            try:
                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment1" not in file:
                    continue

                if "influential_indices" in file:
                    print(file)
                    dataset_id = name[3]
                    if (name[1]=="IS" or name[1]=="WDR"):
                        method = name[1] + "_" + name[2]
                    else:
                        method = name[1]
                    env = name[5]
                    attack_type = name[4]
                    budget = str(np.round(float(name[-4]),2))
                    eps = str(np.round(float(name[-5]),2))
                    key1 = method + "-" + env + "-" + attack_type + "-" + eps + "-" + str(dataset_id) + "-" + budget
                    indices = np.load(full_path)
                    res_indices[key1]=indices
            except:
                pass

        for file in os.listdir(self.path +"/"):
            try:
                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment1" not in file:
                    continue

                elif "before" in file or "after" in file:
                    print(file)
                    dataset_id = name[3]
                    if (name[1]=="IS" or name[1]=="WDR"):
                        method = name[1] + "_" + name[2]
                    else:
                        method = name[1]
                    env = name[5]
                    attack_type = name[4]
                    budget = str(np.round(float(name[-3]),2))
                    eps = str(np.round(float(name[-4]),2))
                    key = method + "-" + env + "-" + attack_type + "-" + eps
                    key1 = method + "-" + env + "-" + attack_type + "-" + eps + "-" + str(dataset_id) + "-" + budget
                    # if float(budget)!=1.2:
                    #     continue
                    indices= res_indices[key1]


                    returns = np.load(full_path).flatten()

                    if flag==True:
                        returns  = returns[indices].flatten()




                    if res.get(key) is None:
                        res[key] = {}
                        res[key]["bw"]={}
                        res[key]["aw"]={}

                    if res[key]["bw"].get(float(budget)) is None:
                        res[key]["bw"][float(budget)]=[]
                    if res[key]["aw"].get(float(budget)) is None:
                        res[key]["aw"][float(budget)]=[]


                    if "before_weights" in file:
                        res[key]["bw"][float(budget)].append(returns)
                    else:
                        res[key]["aw"][float(budget)].append(returns)
            except:
                print(file)
        for key, value in res.items():
            budgets = list(value["bw"].keys())
            max_budget = np.max(budgets)
            bw = value["bw"][max_budget][0]
            aw = value["aw"][max_budget][0]

            sns.displot(bw,  kind="kde", fill=True, color=self.colors[0])
            # g.axes[0].set_xscale('log')
            plt.savefig("plots1/experiments6/"  + key + "_before_wts.pdf")
            plt.clf()
            sns.displot(aw,  kind="kde", fill=True, color=self.colors[1])
            plt.savefig("plots1/experiments6/"  + key + "_after_wts.pdf")
            plt.clf()


    def experiment8(self, res, flag=False):
        results={}
        if flag==False:
            xlabel="Percentage of corrupt points (α)"
            xvals = self.eps_range
            label="_corrupt"
        else:
            xvals = self.budget_range
            label="_budget"
            xlabel="Attacker's budget (frac)"

        for key, value in res.items():
            method = key.split("-")[0].lower()
            env = key.split("-")[1]
            if results.get(env) is None:
                results[env]={}
            if results[env].get(method) is None:
                results[env][method]={}
            results[env][method]["mean"] = self.func(value,0)[0,:]
            # results[env][method]["std"] = np.std(value,0)[0,:]
            results[env][method]["std_lower"], results[env][method]["std_upper"] = get_ci_intervals(value)
            results[env][method]["std_lower"]=  results[env][method]["std_lower"][0,:]
            results[env][method]["std_upper"]=  results[env][method]["std_upper"][0,:]



        for key, value in results.items():
            env = key
            print("experiment 8", env)
            try:
                methods = list(value.keys())
                values=[]
                stds_lower=[]
                stds_upper=[]
                methods2=[]
                for method in methods:
                    if "is_is" in method.lower():
                        methods2.append("WIS")
                    elif "is_pdis" in  method.lower():
                        methods2.append("PDIS")
                    elif "is_cpdis" in method.lower():
                        methods2.append("CPDIS")
                    elif "wdr_wdr" in method.lower():
                        methods2.append("WDR")
                    elif "wdr_dr" in method.lower():
                        methods2.append("DR")
                    else:
                        methods2.append("BRM")
                    values.append(value[method]["mean"])
                    stds_lower.append(value[method]["std_lower"])
                    stds_upper.append(value[method]["std_upper"])


                values_final = np.stack(values,axis=0)
                stds_final_lower = np.stack(stds_lower,axis=0)
                stds_final_upper = np.stack(stds_upper,axis=0)

                if flag==False:
                    add=True
                else:
                    add=False

                ylabel="Percentage error in value function"
                self.plot("experiments8/" + key + "experiment8" + label,values_final,methods2,xvals,xlabel,ylabel,stds_final_lower,stds_final_upper,add=add)
                print("experiment 8")
                plt.clf()
                print(key)
                print(methods)
                for idx in range(len(methods)):
                    print(methods[idx] + "," + str(np.round(values_final[idx,-1],2)) + "," + str(np.round(stds_final_lower[idx,-1],2)) + ","+ str(np.round(stds_final_upper[idx,-1],2)))
                # print(np.round(values_final[:,-1],2))
                # print(np.round(stds_final_lower[:,-1],2))
                # print(np.round(stds_final_upper[:,-1],2))
                plt.clf()
                methods2=[]
                values=[]
                stds_lower=[]
                stds_upper=[]
                for method in methods:
                    if "wdr_dr" not in method.lower() and "is_pdis" not in method.lower():
                        if "is_is" in method.lower():
                            methods2.append("wis")
                        elif "is_pdis" in  method.lower():
                            methods2.append("pdis")
                        elif "is_cpdis" in method.lower():
                            methods2.append("cpdis")
                        elif "wdr_wdr" in method.lower():
                            methods2.append("wdr")
                        elif "wdr_dr" in method.lower():
                            methods2.append("dr")
                        else:
                            methods2.append(method)
                        values.append(value[method]["mean"])
                        stds_lower.append(value[method]["std_lower"])
                        stds_upper.append(value[method]["std_upper"])



                values_final = np.stack(values,axis=0)
                stds_final_lower = np.stack(stds_lower,axis=0)
                stds_final_upper = np.stack(stds_upper,axis=0)
                ylabel="Percentage error in value function"
                self.plot("experiments8/" + key + "subset_experiment8" + label,values_final,methods2,xvals,xlabel,ylabel,stds_final_lower,stds_final_upper,add=add)
                plt.clf()

            except:
                print(key)
                print(methods)
        return results


    def experiment9(self,paths):
        from sklearn.manifold import TSNE
        import pandas as pd
        import seaborn as sns

        # We want to get TSNE embedding with 2 dimensions
        n_components = 2



        for path in paths:
            path1 = "data/"+ path
            original_states =np.load(path1 + "/" + "original_states.npy")
            corrupt_states = np.load(path1 + "/" + "corrupt_states.npy")
            influential_indices = np.load(path1 + "/" + "influential_indices.npy")
            flags = ["non-influential" for x in range(original_states.shape[0])]
            # flags[influential_indices]="influential"
            for x in influential_indices:
                flags[x]="influential"


            tsne = TSNE(n_components)
            tsne_result = tsne.fit_transform(original_states)

            tsne1 = TSNE(n_components)
            tsne_result1 = tsne1.fit_transform(corrupt_states)

            plt.clf()

            # (1000, 2)
            # Two dimensions for each of our images

            # Plot the result of our TSNE with the label color coded
            # A lot of the stuff here is about making the plot look pretty and not TSNE
            tsne_result_df = pd.DataFrame({'dim 1': tsne_result[:,0], 'dim 2': tsne_result[:,1], 'label': flags})
            fig, ax = plt.subplots(1)

            fig = matplotlib.pyplot.gcf()
            # fig.set_size_inches(8.0, 7.0)
            plt.rcParams.update({'font.size': 25})
            plt.rc('legend',fontsize=25)
            plt.xticks(fontsize=25, rotation=0)
            plt.yticks(fontsize=25)

            pal = sns.color_palette("hls",len(tsne_result_df['label'].unique()))
            sns.scatterplot(x='dim 1', y='dim 2', hue='label', data=tsne_result_df, ax=ax,s=120,palette=pal)
            lim = (tsne_result.min()-5, tsne_result.max()+5)
            ax.set_xlim(lim)

            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plt.savefig("plots1/experiments9/"+ path + "_original.pdf")
            plt.clf()

            tsne_result_df1 = pd.DataFrame({'dim 1': tsne_result1[:,0], 'dim 2': tsne_result1[:,1], 'label': flags})
            fig, ax = plt.subplots(1)
            fig = matplotlib.pyplot.gcf()
            # fig.set_size_inches(10.0, 8.0)
            plt.rcParams.update({'font.size': 25})
            plt.rc('legend',fontsize=25)
            plt.xticks(fontsize=25, rotation=0)
            plt.yticks(fontsize=25)
            pal = sns.color_palette("hls",len(tsne_result_df1['label'].unique()))
            sns.scatterplot(x='dim 1', y='dim 2', hue='label', data=tsne_result_df1, ax=ax,s=120,palette=pal)
            lim = (tsne_result1.min()-5, tsne_result1.max()+5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plt.savefig("plots1/experiments9/"+ path + "_corrupt.pdf")


    def experiment10(self, res_new):
        res={}
        res_train={}
        for file in os.listdir(self.path +"/"):
            try:

                full_path = self.path + "/" + file
                name = file.split("_")

                if "experiment8" not in file:
                    continue

                dataset_id = int(name[3])
                if dataset_id not in np.arange(25,30):
                    continue
                method = name[1]
                if (name[1]=="IS" or name[1]=="WDR"):
                    method = name[1] + "_" + name[2]
                env = name[5]
                attack_type = name[4]

                key = method + "-" + env + "-" + attack_type
                key_alt = method + "-" + env + "-" + "influence"
                if "_policy1" in file:
                    returns = np.load(full_path)
                    init_ret = returns[0]
                    tru_rets = returns[1]
                    returns = returns[2:]

                    # num = np.int(np.floor(lent/8))
                    # lent = num*self.budget_range.shape[0]
                    # returns = returns[:lent]
                    if returns.shape[0] < 3*len(self.budget_range):
                        continue
                    rets = returns.reshape(3,-1)
                    if res.get(key_alt) is None:
                        res[key_alt] = []
                    rets = np.abs(100*(rets - init_ret))/(np.abs(init_ret) + 1e-10)

                    res[key_alt].append(rets)

            except:
                print(file)

        for key, value in res.items():
            try:
                if "cartpole" in key and "fqe" in key:
                    print("found")

                value1 = self.func(value,0)[0,:].reshape(1,-1)
                std1_lower, std1_upper = get_ci_intervals(value)
                std1_lower = std1_lower[0,:]
                std1_upper = std1_upper[0,:]

                xlabel="Attacker's budget (frac)"
                ylabel="Percentage error in value function"

                bs = res_new[key]
                value2 = self.func(bs,0)[0,:].reshape(1,-1)
                std2_lower, std2_upper = get_ci_intervals(bs)
                std2_lower = std2_lower[0,:]
                std2_upper = std2_upper[0,:]

                final_val = np.stack((value1,value2),axis=0).squeeze()
                final_std_lower = np.stack((std1_lower,std2_lower),axis=0).squeeze()
                final_std_upper = np.stack((std1_upper,std2_upper),axis=0).squeeze()

                self.plot("experiments10/" + key + "experiment10",final_val,["FSGM Attack","DOPE Attack"],self.budget_range,xlabel,ylabel,final_std_lower,final_std_upper,add=False)
            except:
                pass
        # for key, value in res_train.items():
        #     value = self.func(value,0)
        #     xlabel="attacker's budget"
        #     ylabel="Percentage error in value function (Random Attack)"
        #     self.plot(key + '_experiment4',value,self.projections,self.budget_range,xlabel,ylabel)
        return res


    def process_results(self, res, experiment_id, dir="other_logs"):
        map={"fqe":"BRM","is_is":"WIS","is_pdis":"PDIS","is_cpdis":"CPDIS","wdr_dr":"DR", "wdr_wdr":"WDR"}



        domains = ["cancer","hiv","custom","cartpole","mountaincar"]
        if os.path.exists(dir) == False:
            os.mkdir(dir)
        # if os.path.exists(dir + "/experiment" + str(experiment_id)) == False:
        #     os.mkdir(dir + "/experiment" + str(experiment_id))
        path = dir + "/experiment" + str(experiment_id)+ ".txt"
        file = open(path,'w+')
        if experiment_id==1 or experiment_id==2:
            methods=["fqe","is_is","is_pdis","is_cpdis","wdr_dr","wdr_wdr"]
        else:
            methods=["DOPE Attack","Random DOPE Attack", "Random Attack"]
        for domain in domains:
            value = res[domain]
            key =domain

            for method in methods:
                lower = value[method]["std_lower"]
                upper = value[method]["std_upper"]
                if len(lower)> 10:
                    keys = (np.arange(6))*2
                else:
                    keys = np.arange(5)
                new_lower = lower[keys]
                new_upper = upper[keys]
                new_lower = np.round(new_lower,2)
                new_upper = np.round(new_upper,2)
                str_lower =[str(x) for x in new_lower]
                str_upper =[str(x) for x in new_upper]
                try:
                    method3 = map[method]
                except:
                    method3=method
                r= " & " + method3 +" & "
                for idx in range(len(str_lower)):
                    r=r+ "[" + str_lower[idx] +" , " + str_upper[idx] + "]"
                    if idx == len(str_lower)-1:
                        lent = str(2+len(keys))
                        r=r+"\\\\" +   " \cline{2-" + lent + "}}"
                    else:
                        r=r+" & "

                r=r[:-1] + "\n"
                file.write(r)
        file.flush()
        file.close()

def get_results(res, plotter):
    domains=["hiv"]
    methods=["FQE","IS_is","IS_pdis","IS_cpdis","WDR_dr","WDR_wdr"]
    for domain in domains:
        values = []
        for method in methods:
            key = method + "-" + domain + "-influence"
            value = res[key]
            val = plotter.func(value,0)[0,-1]
            values.append(val)
        print("domain ", domain)
        print(values)








#
#
plotter = Plotter("logs/",axis=0)
res=plotter.experiment1()
get_results(res,plotter)
