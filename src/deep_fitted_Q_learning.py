import sys
sys.path.append("experiments/")
# from argparser_fqe import parse
import numpy as np
# args = parse()
import torch
seed=np.random.randint(1,10)
torch.manual_seed(seed)
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from torch import nn as nn
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)


from utils import TrajectorySet, Transition
from collections import deque
from utils_nn import save_qnet, select_maxq_action
from config import *
import sys
sys.path.append("domains/")
from cancer_env import *
from hiv_env import *
from sepsis_env import *
from cartpole_env import *
from custom import *
from mountaincar_env import *

torch.set_default_tensor_type(torch.DoubleTensor)

from memory import ReplayMemory, Transition

from models import QNet
from train_pipeline import train_pipeline
from utils_nn import load_qnet, error_info
from collections import deque
use_cuda = False #torch.cuda.is_available()
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = DoubleTensor
#400,10
class DeepFittedQIteration():
    """Taken from https://github.com/StanfordAI4HI/RepBM/tree/master/hiv_domain
    FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).
    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions."""
    def __init__(self, env, iterations = 400, K = 10, num_patients = 30, preset_params = None, ins = None,\
        perturb_rate = 0.0, episode_length = 200, cache=False,config=None):
        """Inits the Fitted Q-Iteration planner with discount factor, instantiated model learner, and additional parameters.

        Args:
        model: The model learner object
        gamma=1.0: The discount factor for the domain
        **kwargs: Additional parameters for use in the class.
        """
        self.env = env
        self.cache=True
        self.gamma = env.discount_factor
        print("self.env.discount",self.gamma)
        self.iterations = iterations
        self.K = K
        # Set up regressor
        self.task = env
        self.num_actions = env.action_dim
        self.num_states = env.state_dim
        self.discount_factor = self.gamma
        self.iters=50
        self.scores = deque(maxlen=config.scores_length)
        self.dp_scores = deque(maxlen=config.dp_scores_length)



        self.eps = 1.0
        self.samples = None
        self.preset_params = preset_params
        self.ins = ins
        self.episode_length = episode_length

        self.config = config
        self.qnet = QNet(config.state_dim, config.dqn_hidden_dims, config.action_size)
        self.memory = ReplayMemory(config.buffer_capacity)
        self.epsilon = config.dqn_epsilon

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=config.dqn_alpha)
        self.criterion = torch.nn.MSELoss()

    def encode_action(self, action):
        a = np.zeros(self.num_actions)
        a[action] = 1
        return a

    def reset(self):

        state = self.env.reset()
        return state.flatten()

    def generate_trajectories(self,  sample_num_traj=50,behavior_eps=0.05,debug=False,select_max=False):

        traj_set = TrajectorySet()
        trajectories = []
        scores = deque()
        rewards=[]
        lengths=[]
        for i_episode in range(sample_num_traj):
            trajectory = []

            if i_episode % 10 == 0:
                print("{} trajectories generated".format(i_episode))
            episode, reward = self.run_episode(eps=behavior_eps, track=False,debug=debug,select_max=select_max)
            done = False
            n_steps = 0
            factual = 1
            traj_set.new_traj()
            # acc_isweight = DoubleTensor([1])
            ep_rewards=[]
            gamma=1
            reward_ =0.0
            lengths.append(len(episode))
            while not done:
                action = episode[n_steps][1].flatten()[0]
                state = episode[n_steps][0].flatten()
                next_state = episode[n_steps][3].flatten()
                reward = episode[n_steps][2]
                reward = reward
                # reward_ += gamma*reward
                reward_ += reward
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


        print("lengths",lengths)
        print("mean rewards",np.mean(rewards))


        return trajectories, scores



    def run_episode(self, eps = 0.05, track = False,debug=False, select_max=False, guide_action=False):
        """Run an episode on the environment (and train Q function if modelfree)."""

        # Initialize the environment and state
        state = self.preprocess_state(self.env.reset(),self.config.state_dim)
        done = False
        n_steps = 0
        ep_list=[]
        rewards=0.0

        # print("max length",self.config.max_length)
        # print("select_max",select_max)
        while ((not done) and ( n_steps < self.config.max_length)):

            # Select and perform an action
            if select_max == False:
                action = self.select_action(state, self.qnet, epsilon=eps, action_size=self.config.action_size)
            else:
                action = select_maxq_action(state, self.qnet)
            if guide_action==True :
                action = self.guided_action(state)

            next_state, reward, done = self.env.step(action.item())

            ep_list.append(np.array([state.detach().cpu().numpy().flatten(),action,reward,next_state]))
            rewards +=reward


            # print(next_state.shape, reward, done)
            reward_custom = Tensor([reward])

            if next_state[0] >= 0.5 and self.env.name =="mountaincar_domain":
                reward_custom = Tensor([10])
            elif self.env.name =="mountaincar_domain":
                reward_custom  = Tensor([0])
            next_state = self.preprocess_state(next_state,self.config.state_dim)
            # print(state.shape, next_state.shape, reward.shape, done)
                # Store the transition in memory
            if track==True:
                self.memory.push(state, action, next_state, reward_custom, done, None)
                # Move to the next state
            state = next_state
            n_steps += 1


        return ep_list, rewards

    def guided_action(self,state):
        if random.random() < 0.1:
            return LongTensor([[random.randrange(self.env.action_dim)]])
        if state[0, 1] > 0:
            return LongTensor([[2]])
        else:
            return LongTensor([[0]])


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
        if len(state.shape)==1 or state.shape[0]==1:
            if np.random.rand(1)[0] < eps:
                action = torch.tensor(np.random.randint(0, self.num_actions, state.shape[0])[0])
                probs = softmax(self.qnet(
                    state)+1e-10).reshape(self.env.action_dim)*(1-eps) + eps/self.num_actions
                return action, probs
            else:
                probs = softmax(self.qnet(
                    state)+1e-10).reshape(self.env.action_dim)*(1-eps) + eps/self.num_actions
                return self.qnet(
                    state.type(DoubleTensor)).data.max(1)[1].view(1, 1), probs
        else:
            probs = softmax(self.qnet(
                state)+1e-10).reshape(-1,self.env.action_dim)*(1-eps) + eps/self.num_actions
            return self.qnet(state).max(1)[1].reshape(-1), probs

    def epsilon_decay_per_ep(self,epsilon, config):
        return max(self.config.dqn_epsilon_min, epsilon * self.config.dqn_epsilon_decay)

    def preprocess_state(self,state, state_dim):
        return Tensor(np.reshape(state, [1, state_dim]))

    def select_action(self,state, qnet, epsilon, action_size):
            sample = random.random()
            if sample < epsilon:
                # print("random")
                return LongTensor([[random.randrange(action_size)]])
            else:
                return qnet(
                state.type(Tensor)).data.max(1)[1].view(1, 1)


    def replay_and_optim(self):
        # Adapted from pytorch tutorial example code
        if self.config.dqn_batch_size > len(self.memory):
            return
        transitions = self.memory.sample(self.config.dqn_batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda d: d is False, batch.done)))
        non_final_next_states = torch.cat([t.next_state for t in transitions if t.done is False])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.qnet(state_batch.type(DoubleTensor)).gather(1, action_batch).squeeze()

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.config.dqn_batch_size).type(Tensor)
        next_state_values[non_final_mask] =self.qnet(non_final_next_states.type(Tensor)).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = expected_state_action_values.detach()

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in qnet.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def updatePlan(self,num_episodes=None):
        epsilon = self.epsilon
        if num_episodes is None:
            num_episodes = self.config.dqn_num_episodes
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            guided=False
            if i_episode <= self.config.guide_length  or (random.random()<2*epsilon and self.env.name=="mountaincar_domain"):
                guided=True
            ep_list, reward = self.run_episode(eps=epsilon,guide_action=guided,track=True,select_max=False)
            ep_list, reward = self.run_episode(eps=0.0,guide_action=False)

            # Set the value we want to achieve
            self.dp_scores.append(len(ep_list))
            self.scores.append(reward)
            mean_score = np.mean(self.scores)

            if mean_score >= self.config.expected_scores and i_episode >= self.config.min_episodes:
                print('Ran {} episodes. Solved after {} trials ✔'.format(i_episode, i_episode - self.config.min_episodes))
                print("mean score", mean_score)
                print("expected score", self.config.expected_scores)
                break
            if i_episode % 100 == 0:
                print("mean reward", np.mean(self.scores))
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks. Epsilon is {}'
                      .format(i_episode, mean_score, epsilon))
            # update
            self.replay_and_optim()
            epsilon = self.epsilon_decay_per_ep(epsilon, self.config)

        save_qnet(state={'state_dict': self.qnet.state_dict()},checkpoint="policies",filename=self.env.name + "_qnet.pth.tar")


if __name__ == '__main__':

    num_eps=[100000]
    for num_episodes in num_eps:
        #env = HIVEnv()

        config = hiv_config

        #config = hiv_config
        env = HIVEnv()
        qiter = DeepFittedQIteration(env=env,ins=20,config=config)

        qiter.updatePlan(num_episodes=None)

        eval_qnet = QNet(config.state_dim, config.dqn_hidden_dims, config.action_size)

        #load_qnet(eval_qnet, filename='qnet_mc.pth.tar')
        load_qnet(eval_qnet,checkpoint="policies",filename=env.name + '_qnet.pth.tar')
        eval_qnet.eval()
        qiter.qnet = eval_qnet

        #qiter.qnet = eval_qnet


        print('Learning the tree')
        #qiter.updatePlan()
        #joblib.dump(qiter.tree,'policies/' + env.name + "_" + 'test_extra_tree_gamma_ins' + str(qiter.ins) + '.pkl')
        #qiter.tree = joblib.load('policies/' + env.name + "_" + 'extra_tree_gamma_ins' + str(qiter.ins) + '.pkl')

        #eval_policy.tree = joblib.load('policies/' + "hiv_domain" + "_" + 'extra_tree_gamma_ins' + str(eval_policy.ins) + '.pkl')
        print("Dumped tree")

        trajectories = qiter.generate_trajectories(behavior_eps=0.0,sample_num_traj=50,debug=True)

