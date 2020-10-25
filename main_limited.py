# TODO
# - Let the unconstrained architecture (attend on each other internally )
#   - bottleneck the information going around internally
# - make correlation plot of recon loss and parsing.
# - internal attention effects on parsing.

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

from dataset import Dataset
from vae import VAE
from utils import idxs2string, idxpos2tree

PADDING_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'


def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([model, optimizer], f)


def model_load(fn, device=None):
    global model, optimizer
    with open(fn, 'rb') as f:
        if device is None:
            device = torch.device('cpu' if not args.cuda else 'cuda')
        model, optimizer = torch.load(f, map_location=device)
    return model


###############################################################################
# Training code
###############################################################################

def evaluate(data_iter,
             print_examples=None,
             every=4000,
             score='accuracy',
             print_examples_pos=None):
    # Turn on evaluation mode which disables dropout.
    if hasattr(model, 'module'):
        _model = model.module
    else:
        _model = model
    _model.eval()
    sens_same = 0
    sens_count = 0
    for batch, data in enumerate(data_iter):
        with torch.no_grad():
            nll = _model(data.text).mean()
        if (batch + 1) % every == 0:
            idxs, nodelist = _model.infer(data.text[:, :1])
            print("In: ", idxs2string(data.text[:, 0],
                                      dataset.text_field.vocab.itos))
            print("Out:", idxpos2tree(data.text[:, 0], idxs[0],
                                      nodelist,
                                      dataset.text_field.vocab.itos))
            print()
        mask = (data.text != dataset.text_field.vocab.stoi[PADDING_TOKEN])
        sens_same += -nll * mask.shape[1]
        sens_count += mask.shape[1]
    sens_acc = float(sens_same) / sens_count

    return sens_acc

updates = 0
def train(eval_every=-1, eval_fun=None):
    global updates
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    chunk_count = 0
    start_time = time.time()
    for group in dataset.train_group_iter:
        for data in group:
            loss = model(data.text).mean()
            loss.backward()
            total_loss += loss.detach().data
            chunk_count += 1

        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        optimizer.zero_grad()

        if updates % args.log_interval == 0 and updates > 0:
            elapsed = time.time() - start_time
            if hasattr(model, 'module'):
                _model = model.module
            else:
                _model = model

            print(
                '| epoch {:3d} '
                '| {:5d} batches '
                '| lr {:05.5f} '
                '| beta {:01.3f} '
                '| ms/batch {:5.2f} '
                '| loss {:5.5f}'.format(
                    epoch,
                    updates,
                    optimizer.param_groups[0]['lr'],
                    _model.decoder.beta,
                    elapsed * 1000 / updates,
                    total_loss.item() / chunk_count))
        updates += 1
        if eval_every > 0 and updates % eval_every == 0:
            eval_fun()
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--load-checkpoint', type=str, default='')
    parser.add_argument('--prod-class', type=str, default='Cell',
                        help='model class for generative function')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirection model')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='max sequence length')
    parser.add_argument('--seq_len_test', type=int, default=1000,
                        help='max sequence length')
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nslot', type=int, default=10,
                        help='number of memory slots')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--chunks-per-batch', type=int, default=4)
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropoutm', type=float, default=0.1,
                        help='dropout applied to memory (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropouto', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--enc-word-dropout', type=float, default=0.2,
                        help='Encoder word dropout')

    parser.add_argument('--dec-leaf-dropout', type=float, default=0.1,
                        help='Decoder leaf transform dropout')
    parser.add_argument('--dec-out-dropout', type=float, default=0.1,
                        help='Decoder output transform dropout')
    parser.add_argument('--dec-int-dropout', type=float, default=0.1,
                        help='Decoder attention integration dropout')
    parser.add_argument('--dec-attn-dropout', type=float, default=0.5,
                        help='Decoder attention dropout')

    parser.add_argument('--dec-no-node-attn', action='store_false')
    parser.add_argument('--dec-no-leaf-attn', action='store_false')

    parser.add_argument('--dec-min-depth', type=int, default=8)
    parser.add_argument('--dec-max-depth', type=int, default=30)
    parser.add_argument('--dec-left-discount', type=float, default=0.75)


    parser.add_argument('--encoder-type', type=str, default='OM')
    parser.add_argument('--paren-open', type=str, default='[')
    parser.add_argument('--paren-close', type=str, default=']')
    parser.add_argument('--valid-score', type=str, default='ll')
    parser.add_argument('--test-score', type=str, default='ll')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only')
    parser.add_argument('--logdir', type=str, default='./models/',
                        help='path to save outputs')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--name', type=str, default=randomhash,
                        help='exp name')
    parser.add_argument('--wdecay', type=float, default=0.,
                        help='weight decay applied to all weights')
    parser.add_argument('--std', action='store_true',
                        help='use standard LSTM')
    parser.add_argument('--philly', action='store_true',
                        help='Use philly cluster')
    parser.add_argument('--beta-max', type=float, default=1.)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.logdir, args.name)):
        os.makedirs(os.path.join(args.logdir, args.name))

    ############################################################################
    # Load data
    ############################################################################
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')

    # Compute vocabuary
    dataset = Dataset(
        "PennTreebankParse",
        batch_size=args.batch_size // args.chunks_per_batch,
        sentence_level=True,
        group_size=args.chunks_per_batch,
        vocab_size=10000, preprocess=False,
        device=device
    )

    print(dataset.text_field.vocab.itos[:10])
    model = VAE(
        input_size=args.emsize,
        hidden_size=args.nhid,
        ntokens=len(dataset.text_field.vocab.itos),
        encoder_type=args.encoder_type,
        enc_word_dropout=args.enc_word_dropout,
        enc_dropout=args.dropout,
        enc_dropouti=args.dropouti,
        enc_dropoutm=args.dropoutm,
        dec_prod_class=args.prod_class,
        dec_int_dropout=args.dec_int_dropout,
        dec_leaf_dropout=args.dec_leaf_dropout,
        dec_out_dropout=args.dec_out_dropout,
        dec_attn_dropout=args.dec_attn_dropout,
        nslot=args.nslot,
        beta_max=args.beta_max,
        padding_idx=dataset.text_field.vocab.stoi[PADDING_TOKEN],

        # Depth deciding.
        dec_min_depth=args.dec_min_depth,
        dec_max_depth=args.dec_max_depth,
        dec_left_discount=args.dec_left_discount,
    )
    print(model)

    np.random.seed(args.seed)
    device = torch.device('cpu')
    # model = model.half()
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            print("Let's use", torch.cuda.device_count(), "GPUs?")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = torch.nn.DataParallel(model, dim=1)
    model = model.to(device)



    params = list(model.parameters())
    total_params = sum(np.prod(x.size()) for x in model.parameters())
    print("TOTAL PARAMS: %d" % sum(np.prod(x.size())
                                   for x in model.parameters()))
    print('Args:', args)
    print('Model total parameters:', total_params)

    if not args.test_only:

        print("start training")
        # Loop over epochs.
        lr = args.lr
        stored_acc = float("-inf")
        # At any point you can hit Ctrl + C to break out of training early.

        def eval_test():
            global stored_acc
            print("Evaluating")
            valid_sens_acc = evaluate(dataset.valid_iter,
                                      score=args.test_score,
                                      every=200)
            # test_sens_acc = evaluate(dataset.test_iter,
            #                          print_examples=sys.stdout,
            #                          score=args.test_score,
            #                          every=1e8)
            valid_acc = valid_sens_acc

            print('-' * 89)
            print(
                '| epoch {:3d} '
                '| time: {:5.2f}s '
                '| ↑ valid score: {:.6f} '
                # '| ↑ test acc: {:.4f} '
                ''.format(
                    epoch,
                    (time.time() - epoch_start_time),
                    valid_sens_acc,
                   #  test_sens_acc,
                )
            )
            # if valid_sens_acc >= stored_acc:
            #     model_save(args.model_file)
            #    print('Saving model (new best validation)')
            #     stored_acc = valid_sens_acc
            print('Saving model')
            model_save(args.model_file + ('.%d' % updates))
            print('-' * 89)
            return valid_acc

        try:
            optimizer = None
            # Ensure the optimizer is optimizing params,
            # which includes both the model's weights
            # as well as the criterion's weight (i.e. Adaptive Softmax)
            optimizer = torch.optim.Adam(params,
                                         lr=args.lr,
                                         weight_decay=args.wdecay)
            if args.load_checkpoint != '':
                model_load(args.load_checkpoint)


            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'max', 0.5,
                patience=1, threshold=0,
            )

            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train(eval_every=100, eval_fun=eval_test)
                valid_acc = eval_test()
                scheduler.step(valid_acc)
                if optimizer.param_groups[0]['lr'] < 1e-5:
                    break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    model_load(args.model_file)
    if args.cuda:
        model = model.cuda()
    if args.test_only:
        print(model)

    sens_acc = evaluate(dataset.test_iter,
                        score=args.test_score,
                        every=1e8)
    data = {'args': args.__dict__,
            'parameters': total_params,
            'test_acc': sens_acc}
    print('-' * 89)
    print('| sent acc: {:.4f} ''|\n'.format(sens_acc))
