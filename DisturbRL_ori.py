import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Disturber:
    ''' PPO-Clip '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  
        self.eps = eps  # clip ub
        self.device = device
        self.action_dim = action_dim



    def shuffle(self,states_ori,index):
        states = states_ori.copy()
        states = np.reshape(states,(-1,self.action_dim,3))
        # breakpoint()
        
        states[:,index,:] = states
        # breakpoint()
        return states.reshape(states_ori.shape)

    def _get_shuffle_index(self):
        index=np.array(range(self.action_dim-1))
        np.random.shuffle(index)
        return np.hstack((np.array([0]),index))

    def take_action(self, state,n):
        # n是len(obs),包含自车
        # 这里不涉及梯度计算，所以不shuffle也没有问题

        state = np.array([state])
        # index = self._get_shuffle_index()

        # _state = self.shuffle(state,index)

        state = torch.from_numpy(state).float().to(self.device)

        probs = self.actor(state)

        mask = torch.zeros(probs.shape).to(self.device)
        mask[:,:n]=1


        if torch.isnan(probs).sum() !=0:
            breakpoint()

        probs = probs*mask
        action_dist = torch.distributions.Categorical(probs)

        action = action_dist.sample()

        return action.item()

    def disturb(self, df, action):
        if action == 0:
            return df
        else:
            # print(df.iloc[[action]])
            df = df.drop(index=action, inplace=False)
            
            return df.reset_index(drop=True)



    def update(self, transition_dict,shuffle_n=5):
        actor_loss_list,critic_loss_list=[],[]

        # for i  in range(shuffle_n):
        #     index = self._get_shuffle_index()
            
        #     states = self.shuffle(np.array(transition_dict['states']),index)
        #     states = torch.tensor(states,
        #                         dtype=torch.float).to(self.device)

        #     actions = index[np.array(transition_dict['actions'])]
        #     actions = torch.tensor(actions).view(-1, 1).to(
        #         self.device)

        #     rewards = torch.tensor(np.array(transition_dict['rewards']),
        #                         dtype=torch.float).view(-1, 1).to(self.device)
        #     next_states = self.shuffle(np.array(transition_dict['next_states']),index)

        #     next_states = torch.tensor(next_states,
        #                             dtype=torch.float).to(self.device)
        #     dones = torch.tensor(np.array(transition_dict['dones']),
        #                         dtype=torch.float).view(-1, 1).to(self.device)
        #     td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        #     td_delta = td_target - self.critic(states)

        #     advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
        #                                         td_delta.cpu()).to(self.device)
        #     old_log_probs = torch.log(self.actor(states).gather(1,
        #                                                         actions)).detach()

        #     for _ in range(self.epochs):
        #         log_probs = torch.log(self.actor(states).gather(1, actions))
        #         ratio = torch.exp(log_probs - old_log_probs)
        #         surr1 = ratio * advantage
        #         surr2 = torch.clamp(ratio, 1 - self.eps,
        #                             1 + self.eps) * advantage  # clip
        #         actor_loss = torch.mean(-torch.min(surr1, surr2))  # loss function
        #         critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        #         self.actor_optimizer.zero_grad()
        #         self.critic_optimizer.zero_grad()
        #         actor_loss.backward()
        #         actor_loss_list.append(actor_loss.cpu().detach().item())
        #         critic_loss.backward()
        #         critic_loss_list.append(critic_loss.cpu().detach().item())

        #         self.actor_optimizer.step()
        #         self.critic_optimizer.step()

        index = self._get_shuffle_index()
        
        states = torch.tensor(np.array(transition_dict['states']),
                            dtype=torch.float).to(self.device)

        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(
            self.device)

        rewards = torch.tensor(np.array(transition_dict['rewards']),
                            dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                            dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                            td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # clip
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # loss function
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            actor_loss_list.append(actor_loss.cpu().detach().item())
            critic_loss.backward()
            critic_loss_list.append(critic_loss.cpu().detach().item())
            if torch.isnan(actor_loss).sum() !=0:
                breakpoint()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        return np.array(actor_loss_list).mean(), np.array(critic_loss_list).mean()
    
    def save(self, dir):
        torch.save(self.critic.state_dict(), os.path.join(dir,'critic.pth'))
        torch.save(self.actor.state_dict(), os.path.join(dir,'actor.pth'))

    def load(self,dir):
        self.critic.load_state_dict(torch.load(os.path.join(dir,'critic.pth')))
        self.actor.load_state_dict(torch.load(os.path.join(dir,'actor.pth')))