import torch
import torch.nn as nn
import ordered_memory
import tree_decoder


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, ntokens,
                 encoder_type, enc_dropout, enc_dropouti, enc_dropoutm,
                 enc_word_dropout,
                 dec_prod_class,
                 dec_int_dropout, dec_leaf_dropout,  dec_out_dropout,
                 dec_attn_dropout,
                 dec_min_depth, dec_max_depth, dec_left_discount,
                 nslot, padding_idx, beta_max=1.):
        super(AE, self).__init__()
        self.drop_input = nn.Dropout(enc_dropouti)

        self.decoder = tree_decoder.OrderedMemoryDecoder(
            ntoken=ntokens,
            slot_size=hidden_size,
            producer_class=dec_prod_class,
            padding_idx=padding_idx,
            min_depth=dec_min_depth,
            max_depth=dec_max_depth,
            left_discount=dec_left_discount,
            # leaf_dropout=dec_leaf_dropout,
            # output_dropout=dec_out_dropout,
            integrate_dropout=dec_int_dropout,
            # attn_dropout=dec_attn_dropout,
            # max_depth=nslot,
            # beta_max=beta_max,
        )

        if encoder_type == 'OM':
            self.encoder = ordered_memory.OrderedMemory(
                input_size, hidden_size, nslot,
                ntokens=ntokens,
                padding_idx=padding_idx,
                word_dropout=enc_word_dropout,
                dropout=enc_dropout, dropoutm=enc_dropoutm,
            )
        elif encoder_type == 'birnn':
            self.encoder = ordered_memory.RNNContextEncoder(
                input_size, hidden_size, 5,
                ntokens=ntokens,
                padding_idx=padding_idx,
                dropout=enc_dropout, dropoutm=enc_dropoutm,
            )

    def infer(self, input):
        input = input.t()
        (final_state,
         flattened_internal, flattened_internal_mask,
         rnned_X, X_emb, mask) = self.encoder(input)
        context = (
            flattened_internal, flattened_internal_mask,
            rnned_X, X_emb, mask
        )
        return self.decoder.infer(final_state, context, input)

    def forward(self, input, argmax=True):
        input = input.t()
        (final_state,
         flattened_internal,
         flattened_internal_mask,
         rnned_X, X_emb, mask) = self.encoder(input)
        context = (
            flattened_internal, flattened_internal_mask,
            rnned_X, X_emb, mask
        )
        loss = self.decoder.compute_loss(final_state, context, input)
        return loss, torch.zeros_like(loss)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, seq_len, hidden):
        bsz = hidden.size(0)
        x = hidden.new(seq_len, bsz, self.hidden_size).zero_()
        output, hidden = self.gru(x, hidden[None, :, :])
        output = self.out(output)
        return output

