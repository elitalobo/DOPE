import numpy as np

import os
import sys


class CustomEnv():
    def __init__(self,discount_factor=0.95):
        self.name = "custom"
        self.action_dim = 2
        self.state_dim = 2
        self.gamma = discount_factor
        self.low =[1.0,1.0]
        self.high = [50.0,50.0]
        self.steps=0
        self.reset()
        self.discount_factor = discount_factor



    def reset(self):
        self.state = np.array([1.0,1.0])
        self.steps=0.0
        return self.state

    def step(self,action):


        if action ==0:
            next_state = np.minimum(np.array([0.2,0.45]) + self.state,self.high)   + np.random.rand((2))*1e-5
        else:
            if self.steps > 0:
                flag = np.random.choice([0,1],1,p=[0.05,0.95]) #0.05 works
                if flag:
                    next_state =np.minimum(np.array([0.3,0.5]) + self.state,np.array(self.high)) + np.random.rand((2))*1e-5
                else:
                    next_state = np.array(self.low) + np.random.rand((2))*1e-5
            else:
                next_state = np.minimum(np.array([0.3,0.5]) + self.state,np.array(self.high)) + np.random.rand((2))*1e-5
        reward = self.state[0] + 0.5 *self.state[1]
        self.steps += 1
        self.state = next_state
        done = self.steps==50
        return next_state, reward, done


