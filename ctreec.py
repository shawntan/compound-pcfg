# coding: utf-8
import torch
from torch import nn

DEBUG = False


def exp_safe(x, log_eps, eps):
    safe_mask = x < log_eps
    x = torch.exp(
        x.masked_fill(safe_mask, log_eps)
    ).masked_fill(safe_mask, eps)
    return x


def log_safe(x, log_eps, eps):
    safe_mask = x < eps
    x = torch.log(
        x.masked_fill(safe_mask, eps)
    ).masked_fill(safe_mask, log_eps)
    return x


def leaf_interval(depth, start_idx=0):
    # Leaf interval matrix for in-order
    # Returns: Left exposed, right exposed, adjacency list, end index
    if depth == 0:
        return [start_idx], [start_idx], [], start_idx
    else:
        l_l_side, l_r_side, l_adj_list, last_idx = \
            leaf_interval(depth - 1, start_idx)
        my_idx = last_idx + 1
        r_l_side, r_r_side, r_adj_list, last_idx = \
            leaf_interval(depth - 1, my_idx + 1)
        return (
            [my_idx] + l_l_side,
            [my_idx] + r_r_side,
            [(l, r) for l in l_r_side
             for r in r_l_side] + l_adj_list + r_adj_list,
            last_idx
        )


def extract_label_log_probs(log_probs, labels):
    return log_probs[:, torch.arange(labels.size(1))[:, None], labels.t()]


def index_max(x, out_idx, in_idx):
    v_count = x.size(1)
    val_matrix = torch.full((x.size(0), v_count, v_count),
                            float("-inf"),
                            dtype=torch.long, device=x.device)
    val_matrix[:, out_idx, in_idx] = x[:, out_idx]
    return val_matrix.max(dim=1)

@torch.jit.script
def forward_ctreec(extracted_log_probs, target_lengths,
                   out_uniq_idx, out_uniq_inv, in_idx,
                   start_idxs, end_idxs,
                   eps=torch.tensor(0.),
                   log_eps=torch.tensor(float('-inf'))):

    batch_idx = torch.arange(extracted_log_probs.size(1), dtype=torch.long)
    max_length = extracted_log_probs.size(0)

    acc = torch.zeros_like(extracted_log_probs[0, :, 0])
    prev_probs = torch.zeros_like(extracted_log_probs[0, :])
    prev_probs[:, start_idxs] = torch.tensor(1., dtype=torch.float)

    for t in range(max_length):
        # Keeping it log-safe
        log_prev_probs = log_safe(prev_probs, log_eps, eps)
        log_curr_probs = log_prev_probs + extracted_log_probs[t]

        # Sparse version
        # Extract unique outgoing.
        log_outgoing = log_curr_probs[:, out_uniq_idx]
        # Compute normalisation term only over those.
        log_C = torch.logsumexp(log_outgoing, dim=-1, keepdim=True)
        # Normalise the outgoings, and then re-expand to non-unique (branch out)
        outgoing = exp_safe(log_outgoing - log_C, log_eps, eps)[:, out_uniq_inv]

        end = t + 1 == target_lengths
        if end.any():
            end_instances = torch.nonzero(end)
            end_vals = torch.logsumexp(log_curr_probs[end_instances, end_idxs],
                                       dim=1)
            acc = acc.scatter_add(0, end_instances.flatten(), end_vals)

        if t < max_length - 1:
            mid = t + 1 < target_lengths
            mid_instances = torch.nonzero(mid).flatten()
            mid_vals = log_C[mid_instances, 0]
            acc = acc.scatter_add(0, mid_instances, mid_vals)
            # Transition
            prev_probs = torch.zeros_like(prev_probs)
            prev_probs = prev_probs.index_put(
                (batch_idx[:, None], in_idx[None, :]), outgoing,
                accumulate=True
            )
    return acc


def infer_one(extracted_log_probs, idx_matrix,
              start_idxs, end_idxs):
    neg_inf = torch.tensor(float("-inf"))
    log_M = torch.full_like(extracted_log_probs[0], neg_inf)
    log_M[start_idxs] = extracted_log_probs[0, start_idxs]

    links = []
    t = 0
    for t in range(1, extracted_log_probs.size(0)):
        propagate = torch.cat(
            (log_M, torch.full_like(log_M, neg_inf)), dim=0)[idx_matrix]
        best_log_M, best_link = torch.max(propagate, dim=0)
        log_M = best_log_M + extracted_log_probs[t]
        links.append(best_link)

    end_probs = log_M[end_idxs]
    max_end_val, max_end_pos = torch.max(end_probs, dim=0)
    max_end_pos = end_idxs[max_end_pos]  # convert relative back to absolute

    p = max_end_pos.item()
    reverse_poss = [p]
    while t > 0:
        t = t - 1
        p = links[t][p].item()
        reverse_poss.append(p)
    positions = reverse_poss[::-1]
    return positions


def decode_one(max_log_probs, max_idxs,
               idx_matrix,
               start_idxs, end_idxs,
               max_length=50):
    neg_inf = torch.tensor(float("-inf"))
    log_M = torch.full_like(max_log_probs, neg_inf)
    log_M[start_idxs] = max_log_probs[start_idxs]

    best_score = neg_inf
    best_end_pos = -1
    best_length = -1
    links = []
    for t in range(max_length):
        end_probs = log_M[end_idxs]
        max_end_val, max_end_pos = torch.max(end_probs, dim=0)
        max_end_pos = end_idxs[max_end_pos]  # convert relative back to absolute

        if max_end_val > best_score:
            best_score = max_end_val
            best_end_pos = max_end_pos
            best_length = t

        propagate = torch.cat(
            (log_M, torch.full_like(log_M[:1], neg_inf)),
            dim=0)[idx_matrix]
        best_log_M, best_link = torch.max(propagate, dim=0)
        log_M = best_log_M + max_log_probs
        links.append(best_link)
        if (log_M < best_score).all():
            break
    reverse_poss = [best_end_pos.item()]
    t = best_length
    while t > 0:
        t = t - 1
        best_end_pos = links[t][best_end_pos].item()
        reverse_poss.append(best_end_pos)
    positions = reverse_poss[::-1]
    return max_idxs[positions], positions


class Loss(nn.Module):

    def __init__(self, depth):
        super(Loss, self).__init__()
        start, end, adj, _ = leaf_interval(depth)

        self.depth = depth

        self.log_eps = torch.tensor(-64.)
        self.eps = torch.exp(self.log_eps)
        self.transition = torch.zeros((2 ** (depth + 1) - 1,
                                       2 ** (depth + 1) - 1))
        self.transition[[i for i, _ in adj], [j for _, j in adj]] = 1
        # self.transition = self.transition.to_sparse().t()
        out_idx = torch.tensor([i for i, _ in adj])

        self.out_uniq_idx, self.out_uniq_inv = torch.unique(
            out_idx, return_inverse=True
        )
        self.out_unique_inv = self.out_uniq_inv
        self.in_idx = torch.tensor([j for _, j in adj])
        self.start_idxs = torch.tensor(start, dtype=torch.long)
        self.end_idxs = torch.tensor(end, dtype=torch.long)

        # Hacky hack for sanity.
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                delattr(self, k)
                self.register_buffer(k, v)

        if DEBUG:
            print("transition size:", self.transition.size())

    def forward(self, log_probs, targets, target_lengths,
                extracted_log_probs=None):
        if extracted_log_probs is None:
            extracted = extract_label_log_probs(log_probs, targets)
            extracted_log_probs = extracted.permute(2, 1, 0)
        results = -forward_ctreec(extracted_log_probs, target_lengths.long(),
                                  self.out_uniq_idx, self.out_uniq_inv,
                                  self.in_idx,
                                  self.start_idxs, self.end_idxs,
                                  eps=self.eps, log_eps=self.log_eps)
        if DEBUG:
            print(extracted.size())
            print(results)

        return results

    def decode(self, log_probs):
        log_probs = log_probs.permute(1, 0, 2)
        batch_max_log_probs, batch_max_idxs = torch.max(log_probs, dim=-1)
        out_idx = self.out_uniq_idx[self.out_unique_inv]
        v_count = batch_max_log_probs.size(1)
        idx_matrix = torch.full((v_count, v_count), v_count,
                                dtype=torch.long,
                                device=batch_max_log_probs.device)
        idx_matrix[out_idx, self.in_idx] = out_idx
        batch_results = []
        batch_positions = []
        for i in range(batch_max_log_probs.size(0)):
            idxs, poss = decode_one(
                batch_max_log_probs[i],
                batch_max_idxs[i],
                idx_matrix,
                self.start_idxs, self.end_idxs,
            )
            batch_results.append(idxs)
            batch_positions.append(poss)
        return batch_results, batch_positions

    def infer(self, log_probs, targets, target_lengths,
              extracted_log_probs=None):
        if extracted_log_probs is None:
            extracted = extract_label_log_probs(log_probs, targets)
            extracted= extracted.permute(2, 1, 0)
        out_idx = self.out_uniq_idx[self.out_unique_inv]
        v_count = extracted.size(0)
        idx_matrix = torch.full((v_count, v_count), v_count,
                                dtype=torch.long,
                                device=extracted.device)
        idx_matrix[out_idx, self.in_idx] = out_idx

        batch_positions = []
        for i in range(extracted.size(0)):
            extracted_one = extracted[i, :target_lengths[i], :]
            pos = infer_one(
                extracted_one,
                idx_matrix,
                self.start_idxs, self.end_idxs
            )
            batch_positions.append(pos)
        return batch_positions
