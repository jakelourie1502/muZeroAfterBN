import numpy as np
from global_settings import replay_buffer_size, batch_size, observation_size
from global_settings import training_params
class Replay_Buffer():
    """
    This is a class that can hold the data required for training
    each tuple is :
    obs_t = observed state at that time
    policy_log_t = policy after mcts process
    action_log_t = action chosen, which is a random.choice proprotional to policy.
    reward_log_t+1 = the reward achieved from Ot, At pair.
    done_log_t+1 = whether that Ot, At pair ended the game. note, in our game, reward =1 and done = True happens at the same time.
    fut_val_log_t = 
    """
    def __init__(self):
        self.k = training_params['k']
        self.default_size = batch_size
        self.size = replay_buffer_size
        self.obs = []
        self.policy_log = []
        self.action_log = []
        self.reward_logs = []
        self.done_logs = []
        self.fut_val_logs = []
        self.search_val_logs = []
    
    def add_ep_log(self, metrics):
        """Metrics dictionary of the form
        metrics['obs']
        metrics['policy']
        metrics['action']
        metrics['reward']
        metrics['done']
        metrics['V']
        """
        self.obs.extend(metrics['obs'])
        self.policy_log.extend(metrics['policy'])
        self.action_log.extend(metrics['action'])
        self.reward_logs.extend(metrics['reward'])
        self.done_logs.extend(metrics['done'])
        self.fut_val_logs.extend(metrics['V'])
        self.search_val_logs.extend(metrics['search_value'])
    
    def purge(self):
        no_of_examples = len(self.obs)
        if no_of_examples > self.size:
            reduc = no_of_examples - self.size
            self.obs = self.obs[reduc: ]
            self.policy_log = self.policy_log[reduc: ]
            self.action_log = self.action_log[reduc: ]
            self.reward_logs = self.reward_logs[reduc: ]
            self.done_logs = self.done_logs[reduc: ]
            self.fut_val_logs = self.fut_val_logs[reduc: ]
            self.search_val_logs = self.search_val_logs[reduc: ]
    
    def get_sample_uniform(self, prioritised_sampling = True):
        #### Need to add get sample prioritised.
        
        batch_n = batch_size
        if prioritised_sampling:
            coefs = np.abs(np.array(self.sample_search_val)-np.array(self.fut_val_logs))
            coefs = coefs[:-(self.k+1)]
            coefs = coefs / np.sum(coefs)
            current_length = len(self.obs)-self.k-1 #we don't want one right at the end or it will break.
            indices = np.random.choice(list(range(current_length)),size=batch_n, p=coefs,replace=False)
        else:
            indices = np.random.randint(low = 0, high = len(self.obs), size = batch_n)
        
        # sample_obs = np.array([self.obs[i].reshape(observation_size) for i in indices])
        # sample_policy = np.array([self.policy_log[i] for i in indices])
        # sample_actions = np.array([self.action_log[i] for i in indices])
        # sample_imm_rewards = np.array([self.reward_logs[i] for i in indices])
        # sample_done = np.array([self.done_logs[i] for i in indices])
        # sample_fut_val = np.array([self.fut_val_logs[i] for i in indices])
        # sample_search_val = np.array([self.search_val_logs[i] for i in indices])
        # #### Note: these are all numpy arrays of dimension (m, ...)

        """If you want to have a look at the shapes, types..."""
        # sample_tuples = sample_obs, sample_policy, sample_actions, sample_imm_rewards, sample_fut_val, sample_done sample_search_val, sample_indicies
        # print(sample_tuples[0].shape, sample_tuples[1].shape, sample_tuples[2].shape, sample_tuples[3].shape, sample_tuples[4].shape)
        # print(type(sample_tuples[0]), type(sample_tuples[1]), type(sample_tuples[2]), type(sample_tuples[3]), type(sample_tuples[4]))
        
        
        # return sample_obs, sample_policy, sample_actions, sample_imm_rewards, sample_done, sample_fut_val, sample_search_val, indices
        return indices
