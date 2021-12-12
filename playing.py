import numpy as np
from game_play.frozen_lake import gridWorld

rewards_list = [0,0,1,0,1]


for r in range(len(rewards_list)):
    rewards_list[r] *= 0.96
for r in reversed(range(len(rewards_list)-1)):
    rewards_list[r] += rewards_list[r+1] * 0.96 #THIS GETS VALUE

print(rewards_list)