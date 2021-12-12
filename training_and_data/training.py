import numpy as np 
from global_settings import training_params
from .replay_buffer import Replay_Buffer
import torch

class Training:
    
    def __init__(self,models, replay_buffer):

        self.models = models
        self.representation_model, self.dynamics_model, self.predictions_model = self.models
        self.training_params = training_params
        self.lr = self.training_params['lr']
        self.k = self.training_params['k']
        self.optimizer = self.training_params['optimizer'](self.models, lr=self.lr)
        self.reward_coef = self.training_params['reward_coef']
        self.future_val_coef = self.training_params['future_val_coef']
        self.policy_coef = self.training_params['policy_coef']
        self.replay_buffer = replay_buffer

        def mse(self,a,b):
            return (a-b)**2
        def centropy(self,a,b):
            return np.sum(a * np.log(b))

        def train(self, indices):
            """
            From an observation at time t:
            we want to predict the immediate reward based on action t+k
            and then the future val and policy from being at the state after we've taken k actions.

            If we are only, e.g. 2 observations from the end, we can predict reward 2 steps ahead but not policy / future value because the game is over.
            we work out how many steps ahed we can predict value and policy, and add one to it for predicting reward if the end is due to the game being done.
            we then     
            """
            loss = 0 
            for i in indices: #for each randomly selected indice
                steps_till_done, done_check = self.done_before_k_steps(i)
                p_v_steps = steps_till_done - done_check.astype('int') ### so if done check is false, we reduce the number of p_v_steps we can do.
                observation = torch.tensor(self.replay_buffer.obs[i:i+1])
                rewards = torch.tensor(self.replay_buffer.reward_logs[i:i+steps_till_done])
                future_rewards = torch.tensor(self.replay_buffer.fut_val_logs[i+1:i+p_v_steps+1])
                policies = torch.tensor(self.replay_buffer.fut_val_logs[i+1:i+p_v_steps+1])
                actions = torch.tensor(self.replay_buffer.action_log[i:i+steps_till_done])
                for k in range(steps_till_done): #for each of the subsequent 5 steps (or less if we picked one at the end of a game)

                    #### regardless of k, first step is running the representation model to go from observation to state.
                    state = self.representation_model(observation)
                    #### run dynamic model a number of times depending on which k we're on.
                    for a in range(k):
                        state, _ = self.dynamics_model(state,actions[a]) #put the true actions in and just progress the model.
                    state, reward_prediction = self.dynamics_model(state)
                    if k < p_v_steps:
                        policy_prediction, fut_val_prediction = self.predictions_model(state)
                        loss_policy = self.centropy(policies[k], policy_prediction)                    
                        loss_fut_val = self.mse(future_rewards[k],fut_val_prediction)
                        loss += (self.future_val_coef * loss_fut_val + self.policy_coef * loss_policy))
                    loss_reward = self.mse(rewards[k], reward_prediction)
                    loss += (self.reward_coef * loss_reward)
            
            loss.backward()
            self.optimizer.step(); self.optimizer.zero_grad()
             
        def done_before_k_steps(self,index):
            dones = self.replay_buffer.done_logs[index:index+self.k]
            final_k = 0
            done_check = False
            for t in range(self.k):
                done_check = dones[t]
                final_k += 1
                if done_check == True:
                    break
            return final_k, done_check