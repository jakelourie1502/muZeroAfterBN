import numpy as np
from game_play.frozen_lake import gridWorld

# size = (4,4)
# lakes = [(1,1),(1,3),(2,3),(3,0)]
# goals = {(3,3):1}
# dist = np.zeros((size[0]*size[1]+1))
# dist[0]=1
# env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)


# obs = env.reset()
# print(obs)
# import torch
# obs = torch.tensor(obs)
# print(obs)
# obs = torch.unsqueeze(obs,0)
# print(obs)
# obs, state, reward, done = env.step(1)   
# print(obs)
# import torch
# obs = torch.tensor(obs)
# print(obs)
# obs = torch.unsqueeze(obs,0)
# print(obs)

# rew_list = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-1]
# gamma = 0.98
# for r in reversed(range(len(rew_list)-1)):
#     rew_list[r]+= rew_list[r+1] * gamma
# print(rew_list)
import time
time_now = time.time()
time.sleep(1)
print(time.time()-time_now)