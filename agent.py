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

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

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

    q_vals = self.online_net(states, log=True) 
    q_s_a = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # Q-value of taken action (Q(s_t, a_t)

    with torch.no_grad():

      next_q_vals = self.target_net(next_states).max(1)[0]
      # next_q_vals = nonterminals * next_max_q
      # print(f'next_q: {next_q_vals.shape}')

      # Q-learning: Q_new = Q(s, a) + α[R + γmax_a'Q(s', a') - Q(s, a)]
      q_s_a_prime = returns + (self.discount * next_q_vals)

      # print(q_s_a.shape)
      # print(q_s_a_prime.shape)


    loss = nn.MSELoss()(q_s_a, q_s_a_prime)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      test = (self.online_net(state.unsqueeze(0)))
      print(f'RESULTS 1: {test.shape}')
      # test = test.sum(2)
      return (self.online_net(state.unsqueeze(0))).sum(2).max(1)[0].item()

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

