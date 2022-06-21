import sys
sys.path.append("experiments/")

import torch

seed=10
torch.manual_seed(seed)
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)


from sklearn.ensemble import ExtraTreesRegressor
import joblib
from scipy.special import softmax
from utils import TrajectorySet, Transition
from collections import deque

import sys
sys.path.append("domains/")
from cancer_env import *
from hiv_env import *
from sepsis_env import *
from cartpole_env import *

from mountaincar_env import *


class FittedQIteration():
    """Taken from https://github.com/StanfordAI4HI/RepBM/tree/master/hiv_domain
    FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).
    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions."""
    def __init__(self, env, iterations = 400, K = 10, num_patients = 30, preset_params = None, ins = None,\
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
        self.tree =ExtraTreesRegressor(n_estimators=50, min_samples_split=2, random_state=66)
        self.num_actions = env.action_dim
        self.num_states = env.state_dim
        self.discount_factor = self.gamma
        self.iters=50



        self.eps = 1.0
        self.samples = None
        self.preset_params = preset_params
        self.ins = ins
        self.episode_length = episode_length

    def encode_action(self, action):
        a = np.zeros(self.num_actions)
        a[action] = 1
        return a

    def reset(self):

        state = self.env.reset()
        return state.flatten()

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
                action = int(np.where(episode[n_steps][1] == 1)[0][0])


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



    def run_episode(self, eps = 0.05, track = False,debug=False):
        """Run an episode on the environment (and train Q function if modelfree)."""

        state = self.env.reset()
        # task is done after max_task_examples timesteps or when the agent enters a terminal state
        ep_list = []
        action_list = []
        ep_reward = 0
        iter=0
        done = False
        while not done:
            action = self.policy(state, eps)[0]
            if debug==True:
                print("action",action)
            # print("state",state)
            action_list.append(action)
            next_state, reward, done = self.env.step(action)
            if not track: self.tmp.append(np.hstack([state,action,reward, next_state]))
            else: ep_list.append(np.array([state,self.encode_action(action),reward,next_state,self.ins]))
            state = next_state
            #ep_reward += (reward*self.gamma**self.task.t)
            ep_reward += reward
            iter+=1

            #print(iter)
        if track:
            pass
            # print(np.unique(action_list, return_counts = True),ep_reward)
        return ep_list

    def predictQ(self, state):
        """Get the Q-value function value for the greedy action choice at the given state (ie V(state)).

        Args:
        state: The array of state features

        Returns:
        The double value for the value function at the given state
        """
        Q = [self.tree.predict(np.hstack([state,a*np.ones(len(state)).reshape(-1,1)])) \
        for a in range(self.num_actions)]
        return np.amax(Q,axis=0)

    def policy(self, state,eps = 0.05):
        """Get the action under the current plan policy for the given state.

        Args:
        state: The array of state features


        Returns:
        The current greedy action under the planned policy for the given state. If no plan has been formed,
        return a random action.
        """
        if len(state.shape)==1:
            state= state.reshape(1,-1)
        if len(state.shape)==1 or state.shape[0]==1:
            if np.random.rand(1) < eps:
                action = np.random.randint(0, self.num_actions, state.shape[0])[0]
                probs = np.ones(self.num_actions)*(1.0)/(self.num_actions)
                return action, probs
            else:
                values = self.tree.predict([np.hstack([state.flatten(), a]) for a in range(self.num_actions)])
                return np.argmax(values), softmax(values)
        else:
            if isinstance(state,np.ndarray)==False:
                state = state.detach().cpu().numpy()
            action_arr = (np.ones((state.shape[0],self.num_actions))*np.arange(self.num_actions)).reshape(-1,1)
            state_actions = np.concatenate((np.repeat(state,axis=0,repeats=self.num_actions),action_arr),axis=1)
            values = self.tree.predict(state_actions).reshape(-1,self.num_actions)
            return values.argmax(axis=1), softmax(values,axis=-1)

    def updatePlan(self):
        for k in range(self.K):
            print(k)
            self.tmp = []


            for i in range(self.iters):self.run_episode(eps=self.eps)
            if k==0:
                self.samples = np.vstack(self.tmp)
                self.eps = 0.15
                self.Q = np.zeros(self.samples.shape[0])
                self.tree.fit(self.samples[:,:self.num_states+1],self.Q)
            else: self.samples = np.vstack([self.samples,np.vstack(self.tmp)]);print(self.samples.shape);
            for t in range(self.iterations):
                Qprime = self.predictQ(self.samples[:,-self.num_states:])
                self.Q = self.samples[:,self.num_states+1]+Qprime*self.gamma
                self.tree.fit(self.samples[:,:self.num_states+1],self.Q)
                print("called fit")
                print("t",t)
            print("save Q function at "+str(k)+" epoch")
            joblib.dump(self.tree,'policies/' + self.env.name + "_" + 'extra_tree_gamma_ins'+str(self.ins)+'_K'+str(k)+'.pkl')
            self.run_episode(eps = 0.0, track = True)



if __name__=='__main__':






    env = CartpoleEnv()
    qiter = FittedQIteration(env=env,ins=20)

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






