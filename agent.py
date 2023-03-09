# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

from model import DQN


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.learning_rate = args.learning_rate

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
        q_values = self.online_net(state.unsqueeze(0))
        return q_values.max(1)[1].item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    q_values = self.online_net(states, log=True)[range(self.batch_size), actions]  # Log probabilities log p(s_t, ·; θonline)
    # log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      next_q_values = self.online_net(next_states)
      argmax_indices = next_q_values.argmax(1)

      # Q-learning: Q_new = Q(s, a) + α[R + γmax_a'Q(s', a') - Q(s, a)]
      new_q_values = q_values + self.learning_rate * (returns + self.discount * next_q_values[range(self.batch_size), argmax_indices] - q_values)

    # print(new_q_values.shape)
    # print(q_values.shape)
    loss = nn.MSELoss()(q_values, new_q_values)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

# # -*- coding: utf-8 -*-
# from __future__ import division
# import os
# import numpy as np
# import torch
# from torch import optim
# from torch.nn.utils import clip_grad_norm_
# import torch.nn as nn

# from model import DQN


# class Agent():
#   def __init__(self, args, env):
#     self.action_space = env.action_space()
#     self.batch_size = args.batch_size
#     self.discount = args.discount
#     self.norm_clip = args.norm_clip
#     self.lr = args.learning_rate

#     self.online_net = DQN(args, self.action_space).to(device=args.device)
#     if args.model:  # Load pretrained model if provided
#       if os.path.isfile(args.model):
#         state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
#         if 'conv1.weight' in state_dict.keys():
#           for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
#             state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
#             del state_dict[old_key]  # Delete old keys for strict load_state_dict
#         self.online_net.load_state_dict(state_dict)
#         print("Loading pretrained model: " + args.model)
#       else:  # Raise error if incorrect model path provided
#         raise FileNotFoundError(args.model)

#     self.online_net.train()

#     self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

#   # Acts based on single state (no batch)
#   def act(self, state):
#     with torch.no_grad():
#         q_values = self.online_net(state.unsqueeze(0))
#         return q_values.max(1)[1].item()

#   # Acts with an ε-greedy policy (used for evaluation only)
#   def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
#     return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

#   def learn(self, mem):

#     # Sample transitions
#     idxs, states, actions, rewards, next_states, nonterminals, weights = mem.sample(self.batch_size)

#     # Calculate current state-action values
#     q_values = self.online_net(states)
#     q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

#     with torch.no_grad():

#       # Q-learning: Q_new = Q(s, a) + α[R + γmax_a'Q(s', a') - Q(s, a)]
#       future_q_values = self.online_net(next_states).max(1)[1]
#       # Compute the target
#       new_q_values = q_values + self.lr*(rewards + (self.discount*future_q_values) - q_values)
    
#     # Calculate the loss
#     loss = nn.MSELoss()(q_values, new_q_values)

#     # Optimize the model
#     self.optimiser.zero_grad()
#     loss.backward()
#     clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
#     self.optimiser.step()

#     # Update priorities in the replay buffer
#     mem.update_priorities(idxs, loss.detach().numpy())

#     # Update the target network
#     # if mem.steps % self.target_update == 0:
#     #     self.update_target_net()
  
#   # def update_target_net(self):
#   #   self.target_net.load_state_dict(self.online_net.state_dict())
      
#   # Save model parameters on current device (don't move model between devices)
#   def save(self, path, name='model.pth'):
#     torch.save(self.online_net.state_dict(), os.path.join(path, name))

#   # Evaluates Q-value based on single state (no batch)
#   def evaluate_q(self, state):
#     with torch.no_grad():
#       return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

#   def train(self):
#     self.online_net.train()

#   def eval(self):
#     self.online_net.eval()