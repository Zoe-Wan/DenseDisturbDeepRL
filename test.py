
import os

from DisturbRL import Disturber
import torch


disturber = Disturber(
    state_dim = 33, 
    hidden_dim=128, action_dim=11, 
    actor_lr=1e-3, critic_lr=1e-2, lmbda=0.95, epochs=10, eps=500, gamma=0.98, device=torch.device("cpu"))
disturber.load('./ckptRL/4300')
print([p for p in disturber.actor.parameters()])
# state = [0,-1,0.8,0,0,0.2]
# for i in range(27):
#     state.append(-1)
# action = disturber.take_action(2, state)
# print(action)

