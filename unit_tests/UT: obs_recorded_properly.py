import sys
sys.path.append("..")
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

##### PARAMETER SETTINGS
epochs = 1
workers = 2
max_steps = 10
##### END OF PARAMETER SETTINGS


## LOAD IN PARAMETERS
batch_size = global_settings.batch_size
workers = global_settings.workers
gamma = global_settings.gamma

## INITIALISE MODELS
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
    
    ###### IMPLEMENTATION
    print(replay_buffer.obs)

    ####### END_OF_IMPLEMENTATION
    