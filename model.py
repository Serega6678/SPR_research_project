# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F, init

class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    
    self.fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
    self.fc_v = nn.Linear(args.hidden_size, self.atoms)
    self.fc_a = nn.Linear(args.hidden_size, self.action_space * self.atoms)

    # super().__init__()
    # self.relu = nn.ReLU()
    # self.conv1 = nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0)
    # self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
    # self.fc1 = nn.Linear(3136, hidden_size) 
    # self.fc_v = nn.Linear(hidden_size, 1)
    # self.fc_a = nn.Linear(hidden_size, action_size)
    # # TODO: Distributional version

    # # Orthogonal weight initialisation
    # for name, p in self.named_parameters():
    #   if 'weight' in name:
    #     init.orthogonal(p)
    #   elif 'bias' in name:
    #     init.constant(p, 0)

  def forward(self, x):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    x = F.relu(self.fc1(x))
    v = self.fc_v(x)
    a = self.fc_a(x)
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    return  v + a - a.mean(1, keepdim=True) 
