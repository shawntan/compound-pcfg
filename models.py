import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PCFG import PCFG
from random import shuffle
from ordered_memory import OrderedMemory

class ResidualLayer(nn.Module):
  def __init__(self, in_dim = 100,
               out_dim = 100):
    super(ResidualLayer, self).__init__()
    self.lin1 = nn.Linear(in_dim, out_dim)
    self.lin2 = nn.Linear(out_dim, out_dim)

  def forward(self, x):
    return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class CompPCFG(nn.Module):
  def __init__(self,
               in_vocab = 100,
               out_vocab = 100,
               h_dim = 512, 
               w_dim = 512,
               z_dim = 64,
               state_dim = 256, 
               t_states = 10,
               nt_states = 10,
               inp_padding_idx=0):
    super(CompPCFG, self).__init__()
    self.state_dim = state_dim
    self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
    self.root_emb = nn.Parameter(torch.randn(1, state_dim))
    self.pcfg = PCFG(nt_states, t_states)
    self.nt_states = nt_states
    self.t_states = t_states
    self.all_states = nt_states + t_states
    self.dim = state_dim
    self.register_parameter('t_emb', self.t_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    self.register_parameter('root_emb', self.root_emb)
    self.rule_mlp = nn.Linear(state_dim+z_dim, self.all_states**2)
    self.root_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),                         
                                  nn.Linear(state_dim, self.nt_states))

    self.encoder = OrderedMemory(z_dim, z_dim,
                                 nslot=10, ntokens=in_vocab,
                                 padding_idx=inp_padding_idx)

    self.z_dim = z_dim
    self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   nn.Linear(state_dim, out_vocab))
      
  def enc(self, x):
      (final_state,
       flattened_internal, flattened_internal_mask,
       rnned_X, X_emb, mask) = self.encoder(x)
      return final_state

  def kl(self, mean, logvar):
    result =  -0.5 * (logvar - torch.pow(mean, 2)- torch.exp(logvar) + 1)
    return result

  def forward(self, inp, trg, argmax=False, use_mean=False):
    #x : batch x n
    n = trg.size(1)
    batch_size = trg.size(0)
    z = self.enc(inp)
    self.z = z

    t_emb = self.t_emb
    nt_emb = self.nt_emb
    root_emb = self.root_emb

    root_emb = root_emb.expand(batch_size, self.state_dim)
    t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
    nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)

    if self.z_dim > 0:
      root_emb = torch.cat([root_emb, z], 1)
      z_expand = z.unsqueeze(1).expand(batch_size, n, self.z_dim)
      z_expand = z_expand.unsqueeze(2).expand(batch_size, n, self.t_states, self.z_dim)
      t_emb = torch.cat([t_emb, z_expand], 3)
      nt_emb = torch.cat([nt_emb, z.unsqueeze(1).expand(batch_size, self.nt_states, 
                                                         self.z_dim)], 2)
    root_scores = F.log_softmax(self.root_mlp(root_emb), 1)
    unary_scores = F.log_softmax(self.vocab_mlp(t_emb), 3)
    x_expand = trg.unsqueeze(2).expand(batch_size, n, self.t_states).unsqueeze(3)
    unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
    rule_score = F.log_softmax(self.rule_mlp(nt_emb), 2) # nt x t**2
    rule_scores = rule_score.view(batch_size, self.nt_states, self.all_states, self.all_states)
    log_Z = self.pcfg._inside(unary, rule_scores, root_scores)
    kl = torch.zeros_like(log_Z)
    if argmax:
      with torch.no_grad():
        max_score, binary_matrix, spans = self.pcfg._viterbi(unary, rule_scores, root_scores)
        self.tags = self.pcfg.argmax_tags
      return -log_Z, kl, binary_matrix, spans
    else:
      return -log_Z, kl
