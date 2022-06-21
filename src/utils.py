import matplotlib.pyplot as plt

import sys
sys.path.append("experiments/")
import numpy as np
seed=np.random.randint(1,10)
import torch
torch.manual_seed(seed)
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)
sys.path.append("domains/Cancer/domains/")
sys.path.append("domains/Cancer/")


def softmax_custom( x,t=1e-3):
    if len(x.shape)==1:
        x=x.reshape(1,-1)

    means = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp((x-means)/t)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    return x_exp/(x_exp_sum + 1e-10)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def policy1(state,step):
    return 0


def compute_norm(influences, type):
    if type=="l1":
        influence_n = np.max(np.abs(influences),axis=1)
    elif type=="l2":
        influence_n = np.sum(np.square(influences),axis=1)
    else:
        influence_n = np.sum(np.abs(influences),axis=1)
    return influence_n

def policy2(state,step):
    return 1



class Transition():
    def __init__(self, dictionary):
        self.state = dictionary["state"]
        self.action = dictionary["action"]
        self.next_state = dictionary["next_state"]
        self.reward = dictionary["reward"]
        self.time_step = dictionary["time_step"]



def tensor(x,requires_grad=False):
    try:
        x= x.detach().cpu().numpy()
    except:
        pass
    return torch.tensor(x,requires_grad=requires_grad)

def jacobian(y, x, create_graph=False):
    """

    :param y:
    :param x:
    :param create_graph:
    :return:
    """
    gradient = torch.autograd.grad(y, x, create_graph=True)[0]

    return gradient

def hessian(y, x):
    """

    :param y:
    :param x:
    :return:
    """
    gradient = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]

    hess = []
    for idx in range(gradient.shape[0]):
        hess.append(torch.autograd.grad(gradient[idx], x, retain_graph=True)[0].squeeze())

    hessian = torch.stack(hess, dim=1)
    return hessian, gradient

def tensor(x,type='cpu',requires_grad=True):
    if isinstance(x,np.ndarray):
        x= torch.tensor(x.astype(np.double),requires_grad=requires_grad)
        if type=='cuda':
            x=x.cuda()
        return x


def mixed_derivative(y, x_1, x_2, index):
    """
    computes hessian of y wrt x
    :param y:
    :param x:
    :return:
    """
    gradient = torch.autograd.grad(y, x_1, create_graph=True)[0]
    gradient = gradient[index, :]

    hess = []
    for idx in range(gradient.shape[0]):
        hess.append(torch.autograd.grad(gradient[idx], x_2, retain_graph=True)[0].squeeze())

    mixed_hessian = torch.stack(hess, dim=1)
    return mixed_hessian



def test_hessian_and_mixed_derivative():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    x = tensor(x, requires_grad=True)
    y = tensor(y, requires_grad=True)

    f = torch.sum((3.0 * x - 5.0 * y) ** 2)
    grad_val = jacobian(f, x)
    hess_val = hessian(f, x)
    print("h1", hess_val)
    hess_val = hessian(f, x)
    print("h2", hess_val)
    mixed_val = mixed_derivative(f, x, y)
    grad_expected = 6 * (3.0 * x - 5.0 * y)
    hess_expected = 18 * torch.eye(grad_expected.shape[0])
    mixed_deriv_expected = -30 * torch.eye(grad_expected.shape[0])
    print(grad_val)
    print(hess_val)
    print(mixed_val)
    assert (torch.sum((grad_val == grad_expected) == False) == 0)
    assert (torch.sum((hess_val == hess_expected) == False) == 0)
    assert (torch.sum((mixed_val == mixed_deriv_expected) == False) == 0)

    # test_hessian()


#print(test_hessian())


from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done', 'isweight'))
MyTransition = namedtuple('MyTransition',('state', 'action', 'next_state', 'reward', 'done'
                                          , 'isweight', 'acc_isweight', 'time', 'factual', 'last_factual'
                                          , 'acc_soft_isweight', 'soft_isweight', 'soft_pie', 'pie', 'pib'))
MRDRTransition = namedtuple('MRDRTransition',('state', 'action', 'next_state', 'reward', 'done'
                                              ,'isweight', 'acc_isweight', 'time', 'factual', 'last_factual'
                                              , 'acc_soft_isweight', 'soft_isweight', 'soft_pie', 'pie', 'pib', 'acc_reward', 'wacc_reward'))

# This is a trajectory set, which is used for IS
class TrajectorySet(object):
    def __init__(self, args=None):
        self.trajectories = []
        self.position = -1

    def new_traj(self):
        traj = []
        self.trajectories.append(traj)
        self.position += 1

    def push(self, *args):
        self.trajectories[self.position].append(MyTransition(*args))

    def __len__(self):
        return len(self.trajectories)

# This is the data structure we use to train our MDPnet
# The trick is we store all transition tuples on the same step together, so that when we sample from dataset,
#   we guarantee that the same mini-batch is on the sample time step, which allow us to compute IPM more efficiently
class SampleSet(object):
    def __init__(self, args):
        self.max_len = 400#args.max_length
        self.num_episode = 0
        self.factual = np.zeros(self.max_len)
        self.u = np.zeros(self.max_len) # u_{0:t}
        self.memory = [[] for h in range(self.max_len)]
        self.terminal = []

    def push(self, *args):
        #print(n_steps)
        t = MyTransition(*args)
        self.memory[t.time].append(t)
        if t.factual[0]==1:
            self.factual[t.time] += 1
        if t.done and t.time < self.max_len-1:
            self.terminal.append(t)

    def update_u(self):
        self.num_episode = len(self.memory[0])
        print("mempry",len(self.memory[0]))
        self.u = self.factual/self.num_episode

    def flatten(self):
        self.allsamples = [item for sublist in self.memory for item in sublist]

    def sample(self, batch_size):
        while True:
            time = random.randint(0,self.max_len-1)
            if len(self.memory[time]) >= batch_size:
                return random.sample(self.memory[time], batch_size)

    def sample_terminal(self, batch_size):
        if len(self.terminal) >= batch_size:
            return random.sample(self.terminal, batch_size)
        else:
            return self.terminal

    def flatten_sample(self, batch_size):
        return random.sample(self.allsamples, batch_size)

    def sample_given_t(self, batch_size, time):
        if len(self.memory[time]) >= batch_size:
            return random.sample(self.memory[time], batch_size)
        else:
            return self.memory[time]

    def __len__(self):
        return len(self.memory[0])

# This is the replay memory to train dqn for cartpole - because we need to learn an eval policy at first
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
