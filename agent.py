# -*- coding: utf-8 -*-
from __future__ import division
from kornia.augmentation import RandomCrop
import numpy as np
import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from model import DQN, TransitionModel


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class Agent:
  def __init__(self, args, env, use_augmentations=False):
    self.use_augmentations = use_augmentations

    self.transforms = []
    self.eval_transforms = []

    if self.use_augmentations:
        transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
        eval_transformation = nn.Identity()
        self.transforms.append(transformation)
        self.eval_transforms.append(eval_transformation)

        transformation = Intensity(scale=0.05)
        eval_transformation = nn.Identity()
        self.transforms.append(transformation)
        self.eval_transforms.append(eval_transformation)

    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.lmbd = args.lmbd
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

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

    self.transition_model = TransitionModel(env.action_space())
    self.projection_layer = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024))

    self.online_net.train()
    self.transition_model.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.copy_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.target_encoder_decoder_net = DQN(args, self.action_space).to(device=args.device)
    self.target_encoder_decoder_net.train()
    for param in self.target_encoder_decoder_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  @torch.no_grad()
  def transform(self, images, augment=False):
    images = images.float() / 255. if images.dtype == torch.uint8 else images
    flat_images = images.reshape(-1, *images.shape[-3:])
    processed_images = flat_images
    if augment:
      for transform in self.transforms:
        processed_images = transform(processed_images)
    processed_images = processed_images.view(*images.shape[:-3],
                                             *processed_images.shape[1:])
    return processed_images

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0))[0] * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, initial_states, actions, returns, next_states, nonterminals, weights, all_states, all_actions = mem.sample(self.batch_size)

    # two steps just like in paper
    for _ in range(2):
      # Calculate current state probabilities (online network noise already sampled)
      states = self.transform(initial_states, self.use_augmentations)
      log_ps, states_encoded = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
      log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

      with torch.no_grad():
        # Calculate nth next state probabilities
        pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
        dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
        argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
        self.target_net.reset_noise()  # Sample new target net noise
        pns, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
        pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

        # Compute Tz (Bellman operator T applied to z)
        Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = states.new_zeros(self.batch_size, self.atoms)
        offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

      loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
      mem.update_priorities(idxs, loss.detach().cpu().numpy())

      prev_states = states_encoded
      for step in range(min(self.n, 5)):
        z_cur = self.transition_model(prev_states, all_actions[step])
        prev_states_fl = z_cur.flatten(start_dim=1)

        future_states = self.target_encoder_decoder_net.encode(self.transform(all_states[step + 1]))
        future_states_fl = future_states.flatten(start_dim=1)

        prev_states_fl = self.projection_layer(self.online_net.decode(prev_states_fl))
        future_states_fl = self.target_encoder_decoder_net.decode(future_states_fl).detach()

        loss -= self.lmbd * F.cosine_similarity(prev_states_fl, future_states_fl)

      self.online_net.zero_grad()
      (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
      clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
      clip_grad_norm_(self.transition_model.parameters(), self.norm_clip)  # Clip gradients by L2 norm
      self.optimiser.step()

  def copy_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def update_target_encoder(self):
    if self.use_augmentations:
      theta = 0.0
    else:
      theta = 0.7
    for target_param, online_params in zip(self.target_encoder_decoder_net.convs.parameters(), self.online_net.convs.parameters()):
      target_weight, online_weight = target_param.data, online_params.data
      target_param.data = target_weight * theta + (1 - theta) * online_weight

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0))[0] * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()
    self.transition_model.train()

  def eval(self):
    self.online_net.eval()
    self.transition_model.eval()
