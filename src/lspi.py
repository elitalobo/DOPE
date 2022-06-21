import sys
sys.path.append("experiments/")
seed=10
import torch

torch.manual_seed(seed)
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)

from sklearn.linear_model import LinearRegression,Ridge, LogisticRegression, SGDRegressor
from torch import nn as nn
import pickle
from config import *
import joblib

torch.set_default_tensor_type(torch.DoubleTensor)

from utils import TrajectorySet, Transition
from collections import deque
from feature_expansions import *
import sys
sys.path.append("domains/")
from cancer_env import *
from hiv_env import *
from sepsis_env import *
from cartpole_env import *
use_cuda = False
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = DoubleTensor
from mountaincar_env import *

#400,10
class LSPI():
    """Taken from https://github.com/StanfordAI4HI/RepBM/tree/master/hiv_domain
    FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).
    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions."""
    def __init__(self, env, iterations =200, K = 1000, num_patients = 30, preset_params = None, ins = None,\
        perturb_rate = 0.0, episode_length = 200, cache=False, config=None):
        """Inits the Fitted Q-Iteration planner with discount factor, instantiated model learner, and additional parameters.

        Args:
        model: The model learner object
        gamma=1.0: The discount factor for the domain
        **kwargs: Additional parameters for use in the class.
        """
        self.env = env
        self.cache=True
        self.gamma = env.discount_factor
        self.iterations = iterations
        self.K = K
        # Set up regressor
        self.task = env
        self.tree =Ridge(fit_intercept=False)
        self.num_actions = env.action_dim
        self.num_states = env.state_dim
        self.discount_factor = self.gamma
        self.iters=30



        self.eps = 1.0
        self.samples = None
        self.preset_params = preset_params
        self.ins = ins
        self.episode_length = episode_length

        self.feature_type = config.feature_type
        self.num_centroids = config.num_centroids

        #self.featurizer = GaussianRBFTensor((torch.ones(self.env.state_dim)*10).long(),config.low,config.high)
        self.featurizer = GaussianRBFTensor((torch.ones(self.env.state_dim)*5).long(),env=env)
        self.config = config
        self.fit_featurizer()
        self.weights = None




    def construct_state_action_features(self, states, actions):
            """
            Constructs state-action features from state and actions arrays
            :param states:
            :param actions:
            :param num_actions:
            :return:
            """
            if isinstance(states, np.ndarray):
                states = torch.tensor(states)
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



    def reset(self):

        state = self.env.reset()
        return state.flatten()

    def select_maxq_action(self,state_tensor, qnet):
        return qnet.forward(state_tensor.type(DoubleTensor)).detach().max(1)[1].view(-1, 1)

    def generate_trajectories(self,  sample_num_traj=50,behavior_eps=0.05,debug=False):

        traj_set = TrajectorySet()
        trajectories = []
        scores = deque()
        rewards=[]
        for i_episode in range(sample_num_traj):
            trajectory = []

            if i_episode % 10 == 0:
                print("{} trajectories generated".format(i_episode))
            episode = self.run_episode(eps=behavior_eps, track=True,debug=debug)
            done = False
            n_steps = 0
            factual = 1
            traj_set.new_traj()
            # acc_isweight = FloatTensor([1])
            ep_rewards=[]
            gamma=1
            reward_ =0.0
            while not done:
                action =episode[n_steps][1]


                state = episode[n_steps][0]
                next_state = episode[n_steps][3]
                reward = episode[n_steps][2]
                reward = reward
                reward_ += gamma*reward
                action = action
                ep_rewards.append(gamma*reward)
                gamma = gamma*self.env.discount_factor

                done = n_steps == len(episode) - 1
                traj_set.push(state, action, next_state, reward, done, None, None,
                              n_steps, factual, None, None, None, None, None, None)
                trajectory.append(Transition(state, action, next_state, reward, done,None))
                n_steps += 1
            trajectories.append(trajectory)

            rewards.append(reward_)
            print("episode reward",reward_)

            print(i_episode)



        print("mean rewards",np.mean(rewards))


        return trajectories, scores






    def run_episode(self, eps=0.05, track=False, debug=False, select_max=False, guide_action=False):
        """Run an episode on the environment (and train Q function if modelfree)."""

        # Initialize the environment and state
        state = self.env.reset()
        done = False
        n_steps = 0
        ep_list = []
        rewards = 0.0
        while not done:

            state_tensor = torch.tensor(state)

            # Select and perform an action
            action = self.policy(state_tensor, eps)[0].detach().cpu().numpy()
            if guide_action == True:
                action = self.guided_action(state)[0][0]


            next_state, reward, done = self.env.step(action.item())

            rewards += reward

            # print(next_state.shape, reward, done)
            reward_custom = Tensor([reward])

            if next_state[0] >= 0.5 and self.env.name == "mountaincar_domain":
                reward_custom = Tensor([10])
            elif self.env.name == "mountaincar_domain":
                reward_custom = Tensor([0])
            if not track:
                state = torch.tensor(state)
                state = self.featurizer.transform(state.reshape(1, -1))

                sa_features = self.construct_state_action_features(state, np.array([action]))
                action_tr = torch.tensor(np.array([action])).reshape(-1, 1).double()
                reward_tr = reward_custom.reshape(-1, 1)
                done_tr = torch.tensor(np.array([done])).reshape(-1, 1).double()

                self.tmp.append(
                    torch.cat([sa_features, action_tr, reward_tr, torch.tensor(next_state).reshape(1, -1), done_tr],
                                  dim=1))
            else:
                ep_list.append(np.array([state, action, reward, next_state, self.ins]))
                # Move to the next state
            state = next_state
            n_steps += 1
        if debug==True:
            print("episode length",n_steps)

        return ep_list

    def guided_action(self, state):
        if random.random() < 0.1:
            return LongTensor([[random.randrange(self.env.action_dim)]])
        if state[1] > 0:
            return LongTensor([[2]])
        else:
            return LongTensor([[0]])


    def predictQ(self, state):
        """Get the Q-value function value for the greedy action choice at the given state (ie V(state)).

        Args:
        state: The array of state features

        Returns:
        The double value for the value function at the given state
        """
        if len(state.shape)==1:
            state = state.reshape(1,-1)
        values=[]
        state = self.featurizer.transform(state)
        n = state.shape[0]
        nsa_features = []
        for a in range(self.num_actions):

            sa_features = self.construct_state_action_features(state, np.ones(n)*a)
            weights = self.tree.coef_
            #weights = self.weights
            weights = torch.tensor(weights).reshape(-1, 1)
            value = torch.mm(sa_features, weights)
            values.append(value)
            nsa_features.append(sa_features)
        values = torch.stack(values).squeeze().t()
        actions = torch.argmax(values,dim=1).flatten()
        nsa_features = torch.stack(nsa_features,dim=0)
        final_nsa_features = nsa_features.transpose(0,1)[np.arange(n),actions,:]

        return torch.max(values,dim=1)[0].flatten(),actions, final_nsa_features






    def policy(self, state,eps = 0.05):

        """Get the action under the current plan policy for the given state.

        Args:
        state: The array of state features


        Returns:
        The current greedy action under the planned policy for the given state. If no plan has been formed,
        return a random action.
        """
        softmax = nn.Softmax(dim=1)

        if len(state.shape)==1:
            state= state.reshape(1,-1)
        state = self.featurizer.transform(state)
        if len(state.shape)==1 or state.shape[0]==1:
            if np.random.rand(1) < eps:
                action = np.random.randint(0, self.num_actions, state.shape[0])[0]
                probs = torch.ones(self.num_actions)*(1.0)/(self.num_actions)
                return torch.tensor(action), probs
            else:
                values=[]
                for a in range(self.num_actions):
                    sa_features = self.construct_state_action_features(state.reshape(1,-1),np.array([a]))

                    weights = torch.tensor(self.tree.coef_)
                    value = torch.mm(sa_features,weights.reshape(-1,1))
                    values.append(value)
                values = torch.stack(values)
                return torch.argmax(values.squeeze()), softmax(values.reshape(1,-1))
        else:
            n = state.shape[0]
            values=[]
            for a in range(self.num_actions):
                sa_features = self.construct_state_action_features(state,np.ones(n)*a)
                weights = torch.tensor(self.tree.coef_)
                weights = torch.tensor(weights).reshape(-1, 1)
                value = torch.mm(sa_features, weights)
                values.append(value)
            values = torch.stack(values,dim=1)

            return torch.argmax(values.squeeze(),dim=1), softmax(values.squeeze(),dim=1)

    def fit_featurizer(self):
        num_traj = self.config.num_traj
        ep_lists = [self.run_episode(track=True,eps=1.0) for idx in range(num_traj)]
        states = []
        for traj in ep_lists:
            for transition in traj:
                states.append(torch.tensor(transition[0]))
        states = torch.stack(states)
        tr_states = self.featurizer.transform(states)
        self.state_dim = tr_states.shape[1]




    def epsilon_decay_per_ep(self,epsilon, config):
        return max(self.config.dqn_epsilon_min, epsilon * self.config.dqn_epsilon_decay)


    def updatePlan(self):
        epsilon = self.eps
        for k in range(self.config.dqn_num_episodes):
            print(k)
            self.tmp = []
            guided = False
            if k <= self.config.guide_length or (
                    random.random() < 2 * epsilon and self.env.name == "mountaincar_domain"):
                guided = True

            print("guided",guided)
            for i in range(self.iters):self.run_episode(eps=epsilon,debug=True,guide_action=guided)
            sa_dim = int(self.state_dim * self.num_actions)

            if k==0:
                self.samples = torch.cat(self.tmp)
                #self.Q = torch.zeros(self.samples.shape[0])
                self.weights = torch.rand((sa_dim))
                self.A = torch.zeros((sa_dim,sa_dim))
                self.b = torch.zeros(sa_dim)
                self.Q = np.zeros(self.samples.shape[0])
                self.tree.fit(self.samples[:, :sa_dim], self.Q)


            else:
                self.samples = torch.cat(self.tmp) #torch.cat([self.samples,torch.cat(self.tmp)]);print(self.samples.shape);
            for t in range(self.iterations):
                Qprime, next_actions, next_features = self.predictQ(self.samples[:,-self.num_states-1:-1])
                done = self.samples[:,-1]
                self.Q = self.samples[:,sa_dim+1]+(1-done.flatten())*Qprime.flatten()*self.gamma
                rewards = self.samples[:,sa_dim+1]
                sa_features = self.samples[:,:sa_dim]
                self.tree.fit(self.samples[:,:sa_dim],self.Q)
                # self.A = torch.mm(sa_features.t(),sa_features - self.gamma * next_features)
                # self.b = torch.mm(sa_features.t(),rewards.reshape(-1,1))
                if k==0:
                    break
                #
                # if torch.matrix_rank(self.A)==self.A.shape[1]:
                #     self.weights= torch.mm(torch.inverse(self.A),self.b)
                # else:
                #     self.weights = torch.mm(torch.pinverse(self.A), self.b)
                #

                #self.weights = self.weights + 1e-2* torch.mm(sa_features.t(),(self.Q.flatten()-cur_Q).reshape(-1,1))
                print("weights", np.sum(np.abs(self.tree.coef_)))

                print("called fit")
                print("t",t)
            epsilon = self.epsilon_decay_per_ep(epsilon, self.config)
            self.run_episode(eps = 0.0, track = True,debug=True)


            print("save Q function at "+str(k)+" epoch")
            joblib.dump(self.tree,'policies/' + self.env.name + "_" + 'extra_tree_gamma_ins'+str(self.ins)+'_K'+str(k)+'.pkl')



if __name__=='__main__':






    env = CartpoleEnv()
    config = cartpole_config

    qiter = LSPI(env=env,ins=20,config=config)

    print('Learning the tree')
    qiter.updatePlan()
    joblib.dump(qiter.tree,'policies/' + env.name + "_" + 'test_extra_tree_gamma_ins' + str(qiter.ins) + '.pkl')
    #qiter.tree = joblib.load('policies/' + env.name + "_" + 'extra_tree_gamma_ins' + str(qiter.ins) + '.pkl')

    #eval_policy.tree = joblib.load('policies/' + "hiv_domain" + "_" + 'extra_tree_gamma_ins' + str(eval_policy.ins) + '.pkl')
    print("Dumped tree")

    qiter.generate_trajectories()


    #beh_policy = FittedQIteration(env=env,ins=20)
    #beh_policy.tree = joblib.load('policies/' + "cancer_domain" + "_" + 'extra_tree_gamma_ins' + str(beh_policy.ins) + '.pkl')

    #We assume that behavior policy is eps-greedy with eps=0.05. To sample actions according to behavior policy, call policy(state,eps)






