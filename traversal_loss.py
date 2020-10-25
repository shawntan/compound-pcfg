# coding: utf-8
import torch
from torch import nn
from ctreec import exp_safe, log_safe
DEBUG = False
torch.autograd.set_detect_anomaly(DEBUG)
if DEBUG:
    torch.set_printoptions(precision=2)



def index_max(x, out_idx, in_idx):
    v_count = x.size(1)
    val_matrix = torch.full((x.size(0), v_count, v_count),
                            float("-inf"),
                            dtype=torch.long, device=x.device)
    val_matrix[:, out_idx, in_idx] = x[:, out_idx]
    return val_matrix.max(dim=1)



def sum_product(acc, log_prev_probs, out_uniq_idx, out_uniq_inv, in_idxs,
                batch_idx, log_eps, eps):
    log_outgoing = log_prev_probs[:, out_uniq_idx]
    log_outgoing = log_outgoing + 1e-4 * torch.randn_like(log_outgoing)
    log_C = torch.logsumexp(log_outgoing, dim=-1, keepdim=True)
    acc = acc + log_C[:, 0]
    propagated = exp_safe(log_outgoing - log_C,
                          log_eps, eps)[:, out_uniq_inv]
    combined = torch.zeros_like(log_prev_probs)
    combined = combined.index_put(
        (batch_idx[:, None], in_idxs[None, :]), propagated,
        accumulate=True
    )
    log_combined = log_safe(combined, log_eps, eps)
    return acc, log_combined

def sum_last(log_curr_probs, end_idxs, acc):
    log_last = torch.logsumexp(log_curr_probs[:, end_idxs], dim=1) + acc
    return log_last



def create_forward(propagate_combine=sum_product,
                   tally_final=sum_last):
    def _forward_ctreec(extracted_log_probs, target_lengths,
                        out_uniq_idx, out_uniq_inv, in_idxs,
                        start_idxs, end_idxs,
                        eps, log_eps):
        batch_idx = torch.arange(extracted_log_probs.size(1), dtype=torch.long)
        max_length = extracted_log_probs.size(0)

        acc = torch.zeros_like(extracted_log_probs[0, :, 0])
        result = torch.zeros_like(extracted_log_probs[0, :, 0])
        log_curr_probs = torch.full_like(extracted_log_probs[0, :], log_eps)
        log_prev_probs = log_curr_probs

        for t in range(max_length):
            if t == 0:
                # Initialise
                log_curr_probs[:, start_idxs] = \
                    extracted_log_probs[t, :, start_idxs]

            elif t > 0:
                # Propagate and combine.
                acc, log_combined = propagate_combine(
                    acc, log_prev_probs,
                    out_uniq_idx, out_uniq_inv, in_idxs, batch_idx,
                    log_eps, eps
                )

                log_curr_probs = log_combined + extracted_log_probs[t]
            log_prev_probs = log_curr_probs

            # Save to result if length is done.
            last = t + 1 == target_lengths
            if last.any():
                log_last = tally_final(log_curr_probs, end_idxs, acc)
                result = result.masked_scatter(last, log_last)
        return result
    return _forward_ctreec

accel_forward_ctreec = torch.jit.script(create_forward())
def forward_ctreec(extracted_log_probs, target_lengths,
                   out_idxs, in_idxs,
                   start_idxs, end_idxs,
                   eps=torch.tensor(0.),
                   log_eps=torch.tensor(float('-inf'))):
    out_uniq_idx, out_uniq_inv = torch.unique(out_idxs, return_inverse=True)
    return accel_forward_ctreec(
        extracted_log_probs, target_lengths,
        out_uniq_idx, out_uniq_inv, in_idxs,
        start_idxs, end_idxs,
        eps, log_eps
    )

class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.register_buffer("log_eps", torch.tensor(-64.))
        self.register_buffer("eps", torch.exp(self.log_eps))

        self.positions = []
        def max_hook(grad_output):
            self.positions.insert(0, grad_output[0].nonzero(as_tuple=False).item())

        def max_product(acc, log_prev_probs, out_uniq_idx, out_uniq_inv, in_idxs,
                        batch_idx, log_eps, eps):
            log_prev_probs.register_hook(max_hook)
            v_count = log_prev_probs.size(1)
            val_matrix = torch.full((log_prev_probs.size(0), v_count, v_count),
                                    log_eps,
                                    dtype=torch.float, device=log_prev_probs.device)
            out_idxs = out_uniq_idx[out_uniq_inv]
            val_matrix[:, out_idxs, in_idxs] = log_prev_probs[:, out_idxs]
            log_combined, _ = val_matrix.max(dim=1)
            return acc, log_combined
        def max_last(log_curr_probs, end_idxs, acc):
            log_curr_probs.register_hook(max_hook)
            max_log_prob, _ = log_curr_probs[:, end_idxs].max(dim=1)
            return max_log_prob
        self.max_ctreec = create_forward(propagate_combine=max_product,
                                         tally_final=max_last)


    def forward(self, extracted_log_probs, target_lengths,
                in_idxs, out_idxs, start_idxs, end_idxs):
        extracted_log_probs = extracted_log_probs.permute(2, 1, 0)
        results = -forward_ctreec(extracted_log_probs, target_lengths,
                                  out_idxs, in_idxs,
                                  start_idxs, end_idxs,
                                  eps=self.eps, log_eps=self.log_eps)

        return results

    def infer(self, extracted_log_probs, target_lengths,
              in_idxs, out_idxs, start_idxs, end_idxs):
        results = []
        with torch.set_grad_enabled(True):
            extracted_log_probs = extracted_log_probs.permute(2, 1, 0)
            out_uniq_idx, out_uniq_inv = torch.unique(out_idxs, return_inverse=True)
            for i in range(extracted_log_probs.size(1)):
                self.positions = positions = []
                input_probs = extracted_log_probs[:target_lengths[i], i:i+1]
                backward = self.max_ctreec(
                    input_probs,
                    target_lengths[i:i+1],
                    out_uniq_idx, out_uniq_inv, in_idxs,
                    start_idxs, end_idxs,
                    eps=0., log_eps=float("-inf")
                )
                torch.autograd.grad(backward, input_probs)
                results.append(positions)
        return results
