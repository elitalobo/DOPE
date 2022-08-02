import numpy as np

import os
import sys


class CustomEnv():
    def __init__(self):
        self.action_dim = 2
        self.state_dim = 1
        self.gamma = 0.9
        self.low =1.0
        self.high = 50.0
        self.steps=0
        self.reset()
        self.name="customenv"


    def reset(self):
        self.state = np.array([1.0])
        self.steps=0.0
        return self.state

    def step(self,action):


        if action ==0:
            next_state = np.minimum(0.2 + self.state,self.high)
        else:
            if self.steps > 0:
                flag = np.random.choice([0,1],1,p=[0.05,0.95]) #0.05 works
                if flag:
                    next_state =np.minimum(0.3 + self.state,np.array([self.high]))
                else:
                    next_state = np.array([self.low])
            else:
                next_state = np.minimum(0.3 + self.state,np.array([self.high]))
        reward = self.state[0]
        self.steps += 1
        self.state = next_state
        done = self.steps==50
        return next_state, reward, done


