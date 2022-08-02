import numpy as np

import numpy as np
class PolicyCancer:
    def __init__(self, months_for_treatment=15, eps_behavior=0):
        self.num_actions = 2
        self.months_for_treatment = months_for_treatment
        self.eps_behavior = eps_behavior

    def __call__(self, state, time_step):
        time_step = np.array([time_step]).flatten()
        if len(time_step)==1:
            if np.random.rand() < self.eps_behavior and time_step > 0:
                return np.array([np.random.choice(2)])
            if time_step <= self.months_for_treatment:
                return np.array([1])
            else:
                return np.array([0])
        else:
            actions = np.array(time_step.flatten()< self.months_for_treatment).astype(int)
            return actions.flatten()

    def predict_proba(self, states, time_steps):
        if (len(states.shape))==1:
            states = states.reshape(-1,1)
        probs = np.zeros((states.shape[0],2))
        for idx in range(states.shape[0]):
            if time_steps[idx] <= self.months_for_treatment:
                probs[idx][1]= 1.0 - self.eps_behavior/2
                probs[idx][0] = 1.0 - probs[idx][1]

            else:
                probs[idx][1]=self.eps_behavior/2
                probs[idx][0] = 1.0 - probs[idx][1]
        return probs