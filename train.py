"""Train a model on SQuAD.
"""

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
from models import Seq2Seq, Seq2SeqAttn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

def main(args):
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
    model = Seq2Seq(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    output_size=vocab_size,
                    device=device,
                    drop_prob=args.drop_prob)
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
    # Default project starter code uses Adadelta, but we're going to use Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
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
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                
                train_iter += 1

                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p = model(cw_idxs, qw_idxs)        #(batch_size, q_len, vocab_size)
                
                log_p = log_p.contiguous().view(log_p.size(0) * log_p.size(1), log_p.size(2))
                qw_idxs_target = qw_idxs[:, 1:]     # omitting leading `SOS`
                qw_idxs_target = qw_idxs_target.contiguous().view(qw_idxs_target.size(0) * qw_idxs_target.size(1))
                q_mask = torch.zeros_like(qw_idxs_target) != qw_idxs_target
                q_len = q_mask.sum(-1)
                batch_loss = F.nll_loss(log_p, qw_idxs_target, ignore_index=0, reduction='sum')
                loss = batch_loss / batch_size
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                
                batch_loss_val = batch_loss.item()
                report_loss += batch_loss_val
                cum_loss += batch_loss_val

                tgt_words_num_to_predict = torch.sum(q_len).item()  # omitting leading `<s>`
                report_tgt_words += tgt_words_num_to_predict
                cum_tgt_words += tgt_words_num_to_predict
                report_examples += batch_size
                cum_examples += batch_size

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                
                if train_iter % args.log_every == 0:
                    log.info('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.
                
                # perform validation
                if train_iter % args.valid_niter == 0:
                    log.info('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                            cum_loss / cum_examples,
                                                                                            np.exp(cum_loss / cum_tgt_words),
                                                                                            cum_examples), file=sys.stderr)

                    cum_loss = cum_examples = cum_tgt_words = 0.
                    valid_num += 1

                    print(getWords(cw_idxs[batch_size-1].squeeze().tolist()))
                    print(getWords(qw_idxs[batch_size-1].squeeze().tolist()))
                    util.evaluateRandomly(model, word2Idx, Idx2Word, cw_idxs[batch_size-1].unsqueeze(0), device)

                    print('begin validation ...', file=sys.stderr)

                    # compute dev metrics
                    results = evaluate(model, dev_loader, device, args.use_squad_v2)

                    print('validation: iter %d, dev. ppl %f' % (train_iter, results[args.metric_name]), file=sys.stderr)

                    if saver.is_best(results[args.metric_name]):
                        patience = 0
                        print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                        saver.save(step, model, results[args.metric_name], device)

                        # also save the optimizers' state
                        torch.save(optimizer.state_dict(), model_save_path + '.optim')
                    
                    elif patience < args.patience_limit:
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)

                        if patience == args.patience_limit:
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == args.max_num_trials:
                                print('early stop!', file=sys.stderr)
                                exit(0)

                            # decay lr, and restore from previously best checkpoint
                            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                            model, step = util.load_model(model, args.save_path, args.gpu_ids)

                            print('restore parameters of the optimizers', file=sys.stderr)
                            optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                            # set new lr
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                            # reset patience
                            patience = 0

                    if epoch == args.num_epochs:
                        print('reached maximum number of epochs!', file=sys.stderr)
                        exit(0)
                            
                """
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    #print(getWords(cw_idxs[batch_size-1].squeeze().tolist()))
                    #print(getWords(qw_idxs[batch_size-1].squeeze().tolist()))
                    #util.TeacherForce(model, word2Idx, Idx2Word, cw_idxs[batch_size-1].unsqueeze(0), qw_idxs[batch_size-1].unsqueeze(0), device)
                    #util.evaluateRandomly(model, word2Idx, Idx2Word, cw_idxs[batch_size-1].unsqueeze(0), device)

                    #ema.assign(model)
                    results = evaluate(model,
                                       dev_loader,
                                       device,
                                       args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    #ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')
                """


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

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)
            
            # Forward
            log_p = model(cw_idxs, qw_idxs)        #(batch_size, q_len, vocab_size)
                
            log_p = log_p.contiguous().view(log_p.size(0) * log_p.size(1), log_p.size(2))
            qw_idxs_target = qw_idxs[:, 1:]     # omitting leading `SOS`
            qw_idxs_target = qw_idxs_target.contiguous().view(qw_idxs_target.size(0) * qw_idxs_target.size(1))
            loss = F.nll_loss(log_p, qw_idxs_target, ignore_index=0, reduction='sum')
            nll_meter.update(loss.item(), batch_size)

            q_mask = torch.zeros_like(qw_idxs_target) != qw_idxs_target
            q_len = q_mask.sum(-1)

            # Calculate perplexity        
            cum_loss += loss.item()
            tgt_word_num_to_predict = torch.sum(q_len).item()
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    results_list = [('NLL', nll_meter.avg), \
                ('PPL', ppl)]
    results = OrderedDict(results_list)

    if was_training:
        model.train()

    return results

if __name__ == '__main__':
    main(get_train_args())
