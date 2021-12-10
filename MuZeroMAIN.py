import numpy as np
from game_play.frozen_lake import gridWorld
from game_play.play_episode import Episode
import global_settings
from model.representation import Representation
from model.dynamic import Dynamic
from model.prediction import Prediction
from training_and_data.replay_buffer import Replay_Buffer
from global_settings import epochs
import torch    

### LOAD IN PARAMETERS
batch_size = global_settings.batch_size
workers = global_settings.workers
gamma = global_settings.gamma

### INITIALISE MODELS
representation_model = Representation()
dynamics_model = Dynamic()
predictions_model = Prediction()
replay_buffer = Replay_Buffer()
models = (representation_model, dynamics_model, predictions_model)

for e in range(epochs):
    #### SELF-PLAY
    for m in range(len(models)): 
        models[m].eval()
    for i in range(workers):
        ep = Episode(models) 
        replay_buffer.add_ep_log(ep.play_episode())
    replay_buffer.purge() #keep replay buffer at reasonable size.
    if len(replay_buffer.action_log) > batch_size:
        sample_tuples = replay_buffer.get_sample(batch_size)
        # print(len(sample_tuples))
        # print(sample_tuples[0].shape, sample_tuples[1].shape, sample_tuples[2].shape, sample_tuples[3].shape, sample_tuples[4].shape)
        # print(type(sample_tuples[0]), type(sample_tuples[1]), type(sample_tuples[2]), type(sample_tuples[3]), type(sample_tuples[4]))
    ### TRAINING
    for m in range(len(models)): 
        models[m].train()
