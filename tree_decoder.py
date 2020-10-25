import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import traversal_loss
import math

# from tree_decoder import Cell, update_logbreak
loss = traversal_loss.Loss()

def update_logbreak(prev_log_prob, log_prob):
    return 0.5 * prev_log_prob[:, :, None, 1:] + log_prob


class Cell(nn.Module):
    def __init__(self, input_size, hidden_size, activation,
                 dropout=0.0, branch_factor=2):
        super(Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_hidden_size = 2 * hidden_size

        self.in_linear = nn.Linear(self.input_size, self.cell_hidden_size)
        torch.nn.init.orthogonal_(self.in_linear.weight)
        torch.nn.init.zeros_(self.in_linear.bias)

        out_linear = nn.Linear(self.cell_hidden_size,
                               2 * branch_factor * hidden_size)
        torch.nn.init.orthogonal_(out_linear.weight)
        torch.nn.init.zeros_(out_linear.bias)

        self.output_t = nn.Sequential(
            self.in_linear,
            nn.ReLU(),
            nn.Dropout(dropout),
            out_linear
        )

        self.activation = activation
        self.branch_factor = branch_factor

    def forward(self, x):
        length, batch_size, _ = x.size()
        output_size = (length, batch_size,
                       self.branch_factor,
                       self.hidden_size)

        output = self.output_t(x)
        gates_, cells = output.split((
            self.branch_factor * self.hidden_size,
            self.branch_factor * self.hidden_size
        ), dim=-1)
        gates = torch.sigmoid(
            gates_.view(length, batch_size,
                        self.branch_factor, self.hidden_size))
        cells = cells.view(output_size)

        branches = (gates * self.activation(cells) +
                    (1 - gates) * x[:, :, None, :self.hidden_size])
        return branches



class ReprWrapper(object):
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        if isinstance(self.val, list):
            return "[" + repr(self.val[0]) + " " + repr(self.val[1]) + "]"
        else:
            if self.val is not None:
                return self.val
            else:
                return "NONE!"

    def unwrap(self):
        if isinstance(self.val, list):
            return [self.val[0].unwrap(), self.val[1].unwrap()]
        else:
            return self.val

def flatten_struct(x):
    x = x.permute(0, 2, 1, 3)
    x = x.flatten(0, 1)
    return x

def structen_flat(x):
    x = x.view(-1, 2, x.size(1), x.size(-1))
    x = x.permute(0, 2, 1, 3)
    return x

def drop_masked(x, mask):
    mask = mask.flatten()
    return x[mask]

def subtree_lengths(lengths, expansion):
    struct_lengths = (lengths - 1)[:, :, None, :]
    mask = (struct_lengths > 0)
    mask = mask.expand(-1, -1, 2, -1)
    lengths = torch.ceil(struct_lengths ** expansion[None, None, :, None])
    # print(expansion)
    # print(lengths.flatten())
    return lengths, mask

def remove_eos(X, padding_idx):
    X = X.clone()
    # Remove end
    target_lengths = torch.sum(X != padding_idx, dim=0) - 1
    X[target_lengths,
      torch.arange(X.size(1), dtype=torch.long)] = padding_idx
    X = X[:-1]
    return X, target_lengths


def expand(expand_factor, expansion):
    expand_factor, mask = subtree_lengths(expand_factor, expansion)
    expand_factor = flatten_struct(expand_factor)
    mask = flatten_struct(mask).flatten()
    expand_factor = expand_factor[mask]
    return expand_factor, mask

def build_tree(N, expansion):
    idx = 0
    root = ReprWrapper([None, None])
    nodelist = [root]
    previous_layer = [(N, idx, root)]

    for d in range(N - 1):
        current_layer = []
        for parent_N, _, parent_node in previous_layer:
            if parent_N > 1:
                idx += 1
                left_child = (math.ceil((parent_N - 1) ** expansion[0]), idx,
                              ReprWrapper([None, None]))
                current_layer.append(left_child)
                idx += 1
                right_child = (math.ceil((parent_N - 1) ** expansion[1]), idx,
                               ReprWrapper([None, None]))
                current_layer.append(right_child)

                parent_node.val[0] = left_child[2]
                parent_node.val[1] = right_child[2]
                nodelist.append(left_child[2])
                nodelist.append(right_child[2])

        previous_layer = current_layer
    return nodelist



def recurse(idx_start, n, expand_factor, context, prev_hiddens, fun,
            expansion_factors):
    idxs = torch.arange(idx_start, idx_start + expand_factor.size(0),
                        dtype=torch.long)
    B_l = idxs[:, None].tolist()
    B_r = idxs[:, None].tolist()
    if n > 1:
        prev_expand_factor = expand_factor
        expand_factor, mask = expand(prev_expand_factor, expansion_factors)
        _mask = mask[::2]
        hiddens = fun(prev_hiddens, mask, context[0])
        (expand_factor, desc_hiddens, lower_idxs,
         lower_B_l, lower_B_r, S) = \
            recurse(
                idxs[-1] + 1, n - 1,
                expand_factor,
                context, hiddens, fun,
                expansion_factors
            )
        desc_hiddens.insert(0, hiddens)
        lower_idxs.insert(0, idxs[0])

        j = 0
        for i in range(idxs.size(0)):
            if _mask[i]:
                B_l[i].extend(lower_B_l[j])
                B_r[i].extend(lower_B_r[j + 1])
                for v_i in lower_B_r[j]:
                    for v_j in lower_B_l[j + 1]:
                        S.append((v_i, v_j))
                j = j + 2
    else:
        lower_idxs = [idxs]
        S = []
        desc_hiddens = []
    return expand_factor, desc_hiddens, lower_idxs, B_l, B_r, S


class Operator(nn.Module):
    def __init__(self, cell, output_size, dropout=0.0):
        super(Operator, self).__init__()
        self.cell = cell
        self.leaf_transform = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.cell.hidden_size, 2, bias=True),
        )
        torch.nn.init.zeros_(self.leaf_transform[-1].weight)
        self.in_transform = nn.Sequential(
            nn.Linear(self.cell.hidden_size,
                      self.cell.hidden_size),
        )
        self.global_transform = nn.Sequential(
            nn.Linear(self.cell.hidden_size,
                      self.cell.hidden_size),

        )
        self.out_transform = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.cell.hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)

        self.log_eps = -64.

    def log_leaves(self, hidden):
        logits = self.leaf_transform(hidden)
        # logits[:, :, -1] += 10.
        log_leaf = torch.log_softmax(logits, dim=-1)
        return log_leaf

    def predict(self, prev_hidden):
        branches_ = self.cell(self.dropout(prev_hidden))
        # time_steps, batch_size, branch_factor, hidden_size = branches.size()
        log_leaf_ = self.log_leaves(branches_)
        return branches_, log_leaf_

    def init_parent(self, z):
        prev_hidden = self.in_transform(z)[None, :, :]
        prev_log_leaf = self.log_leaves(
            prev_hidden[:, :, None, :],
        )[:, :, 0, :]
        return prev_hidden, prev_log_leaf

    def forward(self, prev_state, mask, context):
        prev_hidden, prev_log_leaf = prev_state
        prev_hidden = torch.cat((
            prev_hidden,
            context[None, :].expand_as(prev_hidden)), dim=-1)
        branches_, log_leaf_ = self.predict(prev_hidden)
        log_tree_leaf_ = update_logbreak(prev_log_leaf, log_leaf_)

        flat_branches = flatten_struct(branches_)
        flat_leaves = flatten_struct(log_tree_leaf_)
        return flat_branches[mask], flat_leaves[mask]


class OrderedMemoryDecoder(nn.Module):
    def __init__(self, ntoken, slot_size, producer_class,
                 padding_idx=10,  integrate_dropout=0.0,
                 #leaf_dropout, output_dropout, integrate_dropout,
                 #attn_dropout,
                 #node_attention=True, output_attention=False,
                 min_depth=8, max_depth=30, left_discount=0.75):
        super(OrderedMemoryDecoder, self).__init__()
        self.activation = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Tanh(),
        )
        cell = eval(producer_class)
        self.op = Operator(
            cell(input_size=2 * slot_size,
                 hidden_size=slot_size,
                 activation=self.activation),
            output_size=ntoken,
            dropout=integrate_dropout,
        )
        self.padding_idx = padding_idx
        self.beta = 1.
        self.register_buffer('expansion_factors',
                             torch.tensor([left_discount, 1]))
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hidden_size = slot_size


    def forward(self, x, context, depth=10):
        expand_factor = torch.tensor([[[float(depth)]]], dtype=x.dtype).to(x.device)
        prev_hidden, prev_log_leaf = root_state = self.op.init_parent(x)
        _, states, _, B_l, B_r, S = \
            recurse(
                0, depth, expand_factor,
                prev_hiddens=(prev_hidden, prev_log_leaf),
                context=(self.op.global_transform(x),) + context,
                fun=self.op,
                expansion_factors=self.expansion_factors
            )
        states.insert(0, root_state)

        hiddens = torch.cat([h for h, _ in states], dim=0)
        leaves = torch.cat([l for _, l in states], dim=0)
        log_words = self.op.out_transform(hiddens)
        start_idxs = torch.tensor(B_l[0])
        end_idxs = torch.tensor(B_r[0])
        out_idxs = torch.tensor([i for i, _ in S])
        in_idxs = torch.tensor([j for _, j in S])
        return (log_words, leaves,
                start_idxs, end_idxs,
                out_idxs, in_idxs)

    def decide_depth(self, length, min_depth=None, max_depth=None):
        if min_depth is None:
            min_depth = self.min_depth
        if max_depth is None:
            max_depth = self.max_depth
        depth = min(max(min_depth, length), max_depth)
        return depth


    def compute_loss(self, encoding, context, X,
                     min_depth=None, max_depth=None):
        depth = self.decide_depth(X.size(0), min_depth, max_depth)
        target_lengths = torch.sum(X != self.padding_idx, dim=0)
        # X, target_lengths = remove_eos(X, self.padding_idx)
        # print(X)
        log_words, leaves, start_idxs, end_idxs, out_idxs, in_idxs = \
            self.forward(encoding, context, depth=depth)

        log_sel_words = log_words[:, torch.arange(X.size(1))[:, None], X.t()]
        log_leaves = leaves[:, :, :1]
        beta = 1. #  if self.training else 1.
        log_probs = log_sel_words + beta * log_leaves
        batch_losses = loss(
            extracted_log_probs=log_probs,
            target_lengths=target_lengths,
            in_idxs=in_idxs, out_idxs=out_idxs,
            start_idxs=start_idxs, end_idxs=end_idxs
        )
        lengths = target_lengths.to(dtype=encoding.dtype)
        return (batch_losses / lengths)[None, :]

    def infer(self, encoding, context, X,
              min_depth=None, max_depth=None):

        depth = self.decide_depth(X.size(0), min_depth, max_depth)
        target_lengths = torch.sum(X != self.padding_idx, dim=0)
        # X, target_lengths = remove_eos(X, self.padding_idx)
        log_words, leaves, start_idxs, end_idxs, out_idxs, in_idxs = \
            self.forward(encoding, context, depth=depth)

        nodelist = build_tree(depth, self.expansion_factors)
        assert(len(nodelist) == log_words.size(0))
        log_sel_words = log_words[:, torch.arange(X.size(1))[:, None], X.t()]
        log_leaves = leaves[:, :, :1]
        log_probs = log_sel_words +  log_leaves
        idx = loss.infer(
            extracted_log_probs=log_probs,
            target_lengths=target_lengths,
            in_idxs=in_idxs, out_idxs=out_idxs,
            start_idxs=start_idxs, end_idxs=end_idxs
        )
        return idx, nodelist





if __name__ == "__main__":
    omd = OrderedMemoryDecoder(10, 4, "Cell")
    enc = torch.randn(1, 4)
    X = torch.randint( low=0, high=10, size=(15, 1))
    omd.compute_loss(enc, None, X)

