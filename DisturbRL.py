import torch
import torch.nn as nn
import numpy as np
from numpy import inf
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import global_val
import random
import math
import os
# 使用mc first visit估计方法
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        # self.weights = []
        self.ndd_prossi_list = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        # del self.weights[:]
        del self.ndd_prossi_list[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, mask):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist_mask = torch.zeros(action_probs.shape).to(device)
            dist_mask[mask] = 1
            action_probs*=dist_mask
            dist = Categorical(action_probs)

        # action = torch.argmax(action_probs)
        # action = dist.sample()

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    def act_test(self, state,mask):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            breakpoint()
            dist_mask = torch.zeros(action_probs.shape).to(device)
            dist_mask[mask] = 1
            action_probs*=dist_mask

            dist = Categorical(action_probs)
        # action = torch.argmax(action_probs)
        action = dist.sample()

        action_logprob = dist.log_prob(action)

        state_val = self.critic(state)

        return action.detach(),action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class Disturber:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    # def disturb(self, df, action):
    #     if action == 0:
    #         return df
    #     else:
    #         df = df.drop(index=action, inplace=False)
    #         return df.reset_index(drop=True)
    
    def disturb(self, df, actions, indexes):

        ndf=df.copy()
        disturb_actions = []
        for ori_i in range(6):
            i = indexes[ori_i]
            a = actions[ori_i]
            if i is not None:
                if a==0:
                    ndf.loc[i,'x']+=10
                elif a==1:
                    ndf.loc[i,'x']-=10
                elif a==2:
                    ndf.loc[i,'v']+=5
                elif a==3:
                    ndf.loc[i,'v']-=5
                elif a==4:
                    ndf.drop(i)

                elif a == 6:
                    pass
                disturb_actions.append(a)
            else:
                if a==5:
                    egox = ndf.loc[0,'x']+30
                    egov = ndf.loc[0,'v']
                    egol = ndf.loc[0,'lane_id']
                    new_row = {'x':egox,'v':egov, 'lane_id':egol}
                    ndf.loc[len(ndf)-1] = new_row
                    disturb_actions.append(5)
                else:
                    disturb_actions.append(6)
        return ndf,disturb_actions
            
    def take_action(self, state,mask):


        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state,mask)
        
        if action!=36:
            if random.random()<0.1:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
                return action.item(),state_val,(action_logprob).detach().cpu().numpy()
                
            else:
                return 36,None,0
        else:
            return 36,None,0



    def take_action_test(self, state,mask):


        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act_test(state,mask)
            if action!= 36:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
                return action.item(), state_val,action_logprob.detach().cpu().numpy()
            else:
                return 36,None,0

    def update(self):
        # Monte Carlo estimate of returns
        actor_loss_list,critic_loss_list=[],[]
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)


            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO 这里为什么要合并在一起？？
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            loss = actor_loss + critic_loss - 0.01 * dist_entropy
            actor_loss_list.append(actor_loss.cpu().detach().numpy())
            critic_loss_list.append(critic_loss.cpu().detach().numpy())
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        


        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        

        return np.array(actor_loss_list).mean(), np.array(critic_loss_list).mean()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), os.path.join(checkpoint_path,'policy.pth'))
   
    def load(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path,'policy.pth')
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

