import numpy as np
from global_settings import replay_buffer_size, batch_size, observation_size
class Replay_Buffer():
    """
    This is a class that can hold the data required for training
    for time stamp t-1, data required is
    Observation_t-1
    for t in range (t: t+5):
        (policy post MCTS,
        action,
        future value of state,
        immedate reward
        )
    Also needs functions that:
    1) add a dictionary of metrics
    1) Purge (keep it to it's maximum size)
    2) Extract a sample (initially just a random sample)
    """
    def __init__(self):
        self.default_size = batch_size
        self.size = replay_buffer_size
        self.obs = []
        self.policy_log = []
        self.action_log = []
        self.reward_logs = []
        self.fut_val_logs = []
    
    def add_ep_log(self, metrics):
        """Metrics dictionary of the form
        metrics['obs']
        metrics['policy']
        metrics['action']
        metrics['reward']
        metrics['V']
        """
        self.obs.extend(metrics['obs'])
        self.policy_log.extend(metrics['policy'])
        self.action_log.extend(metrics['action'])
        self.reward_logs.extend(metrics['reward'])
        self.fut_val_logs.extend(metrics['V'])
    
    def purge(self):
        no_of_examples = len(self.obs)
        if no_of_examples > self.size:
            reduc = no_of_examples - self.size
            self.obs = self.obs[reduc: ]
            self.policy_log = self.policy_log[reduc: ]
            self.action_log = self.action_log[reduc: ]
            self.reward_logs = self.reward_logs[reduc: ]
            self.fut_val_logs = self.fut_val_logs[reduc: ]
    
    def get_sample(self, size=None):
        if size == None:
            batch_n = batch_size
        else:
            batch_n = size
        indices = np.random.randint(low = 0, high = len(self.obs), size = batch_n)
        sample_obs = np.array([self.obs[i].reshape(observation_size) for i in indices])
        sample_policy = np.array([self.policy_log[i] for i in indices])
        sample_actions = np.array([self.action_log[i] for i in indices])
        sample_imm_rewards = np.array([self.reward_logs[i] for i in indices])
        sample_fut_val = np.array([self.fut_val_logs[i] for i in indices])
        #### Note: these are all numpy arrays of dimension (m, ...)

        """If you want to have a look at the shapes, types..."""
        # sample_tuples = sample_obs, sample_policy, sample_actions, sample_imm_rewards, sample_fut_val
        # print(sample_tuples[0].shape, sample_tuples[1].shape, sample_tuples[2].shape, sample_tuples[3].shape, sample_tuples[4].shape)
        # print(type(sample_tuples[0]), type(sample_tuples[1]), type(sample_tuples[2]), type(sample_tuples[3]), type(sample_tuples[4]))
        
        
        return sample_obs, sample_policy, sample_actions, sample_imm_rewards, sample_fut_val, indices

