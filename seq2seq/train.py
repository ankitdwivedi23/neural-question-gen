"""Train a model on SQuAD.

code adapted from:
    > https://github.com/chrischute/squad
"""

from typing import Union

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import json
import sys
import time
import math
import os

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import Seq2SeqGru, Seq2Seq, Seq2SeqAttn, TransformerModel
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

PAD = 0
SOS = 2
EOS = 3

args = get_train_args()

# Set up logging and devices
args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
log = util.get_logger(args.save_dir, args.name)
device, args.gpu_ids = util.get_available_devices()
log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
args.batch_size *= max(1, len(args.gpu_ids))

# Set random seed
log.info(f'Using random seed {args.seed}...')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)    

# Get embeddings
log.info('Loading embeddings...')
word_vectors = util.torch_from_json(args.word_emb_file)

log.info('Loading word2Idx...')
word2Idx = json.loads(open(args.word2idx_file).read())
Idx2Word = {v: k for (k,v) in word2Idx.items()}

vocab_size = word_vectors.size(0)
print(f"Vocab size: {vocab_size}")

def getWords(idxList):
        words = []
        for i in idxList:
            words.append(Idx2Word[i])
        return words

def create_new_model():
        if args.model_type == "seq2seq":
            return Seq2Seq(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    output_size=vocab_size,
                    device=device,
                    drop_prob=args.drop_prob,
                    num_layers=args.num_layers)
        elif args.model_type == "seq2seq_attn":
            return Seq2SeqAttn(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    output_size=vocab_size,
                    device=device,
                    drop_prob=args.drop_prob,
                    num_layers=args.num_layers)

def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
    * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
    * Values are not interpolated, which corresponds to
    ``numpy.percentile(..., interpolation="nearest")``.
    
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k)
    return result

def main():
    #torch.set_default_dtype(torch.float64)
    
    #  Get model
    log.info('Building model...')
    model = create_new_model()    
    model = nn.DataParallel(model, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 args.best_model_name,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    model_save_path = os.path.join(args.save_dir, args.best_model_name)

    # Get optimizer and scheduler    
    # Default project starter code uses Adadelta, but we're going to use SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr))
    scheduler = sched.MultiStepLR(optimizer, milestones=[i for i in range(args.epoch_start_decay, args.num_epochs + 1)], gamma=0.5)

    num_trial = 0
    train_iter = train_iter_actual = batch_size_actual = batch_loss = patience = total_loss = report_loss = batch_words = total_words = report_words = 0
    total_examples = report_examples = epoch = valid_num = 0
    train_time = begin_time = time.time()

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    optimizer.zero_grad()
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        scheduler.step()
        log.info(f"Starting epoch {epoch}, with learning rate {optimizer.param_groups[0]['lr']}")

        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, re_cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                
                train_iter += 1

                cw_idxs = cw_idxs.to(device)
                re_cw_idxs = re_cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                minibatch_size = cw_idxs.size(0)
                batch_size_actual += minibatch_size

                # Setup for forward
                src_idxs = re_cw_idxs
                #copy_idxs = torch.cat((torch.zeros((minibatch_size, 1), device=device, dtype=torch.long), src_idxs, torch.zeros((minibatch_size, 1), device=device, dtype=torch.long)), dim=-1)
                #copy_idxs[:,0] = SOS
                #copy_idxs[:,-1] = EOS
                tgt_idxs = qw_idxs[:, :-1]
                tgt_idxs_y = qw_idxs[:, 1:]
                
                src_mask = src_idxs != PAD
                tgt_mask = tgt_idxs != PAD

                # Forward
                log_p = model(src_idxs, tgt_idxs)                  #(batch_size, q_len, vocab_size)

                # Print for debugging              
                '''
                print("Train Context:")
                print(src_idxs[0])
                print(getWords(src_idxs[0].tolist()))
                print("Train Question:")
                print(tgt_idxs[0])
                print(getWords(tgt_idxs[0].tolist()))
                print("Train Predicted:")
                print(log_p[0].argmax(-1))
                print(getWords(log_p[0].argmax(-1).tolist()))
                '''
                log_p = log_p.contiguous().view(-1, log_p.size(-1))

                tgt_idxs_y = tgt_idxs_y.contiguous().view(-1)
                tgt_no_pad = tgt_idxs_y != PAD
                tgt_len = tgt_no_pad.sum(-1)

                minibatch_words = torch.sum(tgt_len).item()
                batch_words += minibatch_words
                
                minibatch_loss = F.nll_loss(log_p, tgt_idxs_y, ignore_index=0, reduction='sum')
                loss = minibatch_loss / minibatch_size
                loss.backward()

                batch_loss += minibatch_loss.item()          

                # Backward
                if train_iter % 4 == 0 or minibatch_size < args.batch_size:                                        
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    report_loss += batch_loss
                    total_loss += batch_loss
                    optimizer.zero_grad()                    

                    # Log info
                    step += batch_size_actual
                    progress_bar.update(batch_size_actual)
                    progress_bar.set_postfix(epoch=epoch,
                                            NLL=batch_loss)
                    
                    
                    report_words += batch_words
                    total_words += batch_words
                    report_examples += batch_size_actual
                    total_examples += batch_size_actual

                    batch_words = 0
                    batch_loss = 0
                    batch_size_actual = 0
                    train_iter_actual += 1
                
                    if train_iter_actual % args.log_every == 0:
                        
                        # Print for debugging
                        '''

                        print("Context Words:")
                        print(getWords(src_idxs[0].squeeze().tolist()))                    
                        
                        print("Question Words:")
                        print(getWords(tgt_idxs[0].squeeze().tolist()))

                        print("Predicted Words:")
                        model.eval()
                        predicted_idxs = util.greedyDecode(model, word2Idx, Idx2Word, src_idxs[0].unsqueeze(0), device)
                        print(predicted_idxs)
                        print(getWords(predicted_idxs))
                        model.train()
                        '''
                        
                        log.info('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                        'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter_actual,
                                                                                            report_loss / report_examples,
                                                                                            math.exp(report_loss / report_words),
                                                                                            total_examples,
                                                                                            report_words / (time.time() - train_time),
                                                                                            time.time() - begin_time))

                        train_time = time.time()
                        report_loss = report_words = report_examples = 0.

                    # perform validation
                    if args.valid_niter > 0 and train_iter_actual % args.valid_niter == 0:
                        log.info('epoch %d, iter %d, totat loss %.2f, total ppl %.2f total examples %d' % (epoch, train_iter_actual,
                                                                                                total_loss / total_examples,
                                                                                                np.exp(total_loss / total_words),
                                                                                                total_examples))

                        total_loss = total_examples = total_words = 0.
                        valid_num += 1

                        print('begin validation ...', file=sys.stderr)

                        # compute dev metrics
                        results = evaluate(model, dev_loader, device, args.use_squad_v2)

                        log.info('validation: iter %d, dev. NLL %f, dev. ppl %f' % (train_iter, results['NLL'], results['PPL']))

                        # code for early stopping, not used in the training of our models
                        '''
                        if saver.is_best(results[args.metric_name]):
                            patience = 0
                            log.info('save currently the best model to [%s]' % model_save_path)
                            saver.save(step, model, results[args.metric_name], device)

                            # also save the optimizers' state
                            torch.save(optimizer.state_dict(), model_save_path + '.optim')
                        
                        elif patience < args.patience_limit:
                            patience += 1
                            log.info('hit patience %d' % patience)

                            if patience == args.patience_limit:
                                num_trial += 1
                                log.info('hit #%d trial' % num_trial)
                                if num_trial == args.max_num_trials:
                                    log.info('early stop!')
                                    exit(0)

                                # decay lr, and restore from previously best checkpoint
                                lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                                log.info('load previously best model and decay learning rate to %f' % lr)

                                model, step = util.load_model(model, model_save_path, args.gpu_ids)

                                log.info('restore parameters of the optimizers')
                                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                                # set new lr
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = lr

                                # reset patience
                                patience = 0
                        '''
            if epoch == args.num_epochs:
                log.info('reached maximum number of epochs!')
                log.info('save the model to [%s]' % model_save_path)
                saver.save(step, model, results[args.metric_name], device)
                exit(0)

def evaluate(model, data_loader, device, use_squad_v2):
    """ Evaluate on dev questions
    @param model (Module): Question Generation Model
    @param data_loader (DataLoader): DataLoader to load dev examples in batches
    @param device (string): 'cuda:0' or 'cpu'
    @param use_squad_v2 (bool): boolean flag to indicate whether to use SQuAD 2.0 
    @returns results (dictionary of on dev questions)
    """
    nll_meter = util.AverageMeter()

    was_training = model.training
    model.eval()

    total_loss = 0.
    total_words = 0.
    total_examples = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for cw_idxs, re_cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            
            cw_idxs = cw_idxs.to(device)
            re_cw_idxs = re_cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)
            
            # Setup for forward
            src_idxs = re_cw_idxs
            #copy_idxs = torch.cat((torch.zeros((batch_size, 1), device=device, dtype=torch.long), src_idxs, torch.zeros((batch_size, 1), device=device, dtype=torch.long)), dim=-1)
            #copy_idxs[:,0] = SOS
            #copy_idxs[:,-1] = EOS
            tgt_idxs = qw_idxs[:, :-1]
            tgt_idxs_y = qw_idxs[:, 1:]

            # Forward
            log_p = model(src_idxs, tgt_idxs)        #(batch_size, q_len, vocab_size)

            # Print for debugging            
            print("Validation Context:")
            print(src_idxs[0])
            print(getWords(src_idxs[0].tolist()))
            print("Validation Question:")
            print(tgt_idxs[0])
            print(getWords(tgt_idxs[0].tolist()))
            print("Validation Predicted:")
            print(log_p[0].argmax(-1))
            print(getWords(log_p[0].argmax(-1).tolist()))

            log_p = log_p.contiguous().view(-1, log_p.size(-1))
            tgt_idxs_y = tgt_idxs_y.contiguous().view(-1)
            tgt_no_pad = tgt_idxs_y != PAD
            tgt_len = tgt_no_pad.sum(-1)

            batch_words = torch.sum(tgt_len).item()
            total_words += batch_words
            total_examples += batch_size
            
            batch_loss = F.nll_loss(log_p, tgt_idxs_y, ignore_index=0, reduction='sum')
            loss = batch_loss / batch_size
            nll_meter.update(loss.item(), batch_size)

            # Calculate perplexity        
            batch_loss_val = batch_loss.item()
            total_loss += batch_loss_val

        ppl = np.exp(total_loss / total_words)

    results_list = [('NLL', nll_meter.avg), \
                ('PPL', ppl)]
    results = OrderedDict(results_list)

    if was_training:
        model.train()

    return results

#########################################################################################################

def gruMain(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    log.info('Loading word2Idx...')
    word2Idx = json.loads(open(args.word2idx_file).read())
    Idx2Word = {v: k for (k,v) in word2Idx.items()}
    vocab_size = len(word2Idx)

    def getWords(idxList):
        words = []
        for i in idxList:
            words.append(Idx2Word[i])
        return words

    # Get model
    log.info('Building model...')
    model = Seq2SeqGru(Idx2Word=Idx2Word, hidden_size=args.hidden_size,
                    output_size=vocab_size,
                    device=device)
    model = model.to(device)

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    epoch = 0
    report_loss = 0
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, re_cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                re_cw_idxs = re_cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = re_cw_idxs.size(0)
                
                c_mask = torch.zeros_like(re_cw_idxs) != re_cw_idxs
                q_mask = torch.zeros_like(qw_idxs) != qw_idxs

                # Forward

                batch_loss = model(re_cw_idxs, re_cw_idxs[:, 0:1])
                loss = batch_loss / batch_size

                # Evaluate on Train
                if epoch == args.num_epochs-1:
                    for i in range(batch_size):
                        idx = re_cw_idxs[i]
                        print("Expected : " + str(getWords([idx[0:1].squeeze().tolist()])))
                        print(model.evaluate(idx.unsqueeze(0)))

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss)

        if epoch == args.num_epochs:
            log.info('reached maximum number of epochs!')


if __name__ == '__main__':
    main()