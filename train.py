"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
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

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import Seq2Seq
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
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
    vocab_size = len(word2Idx)

    # Get model
    log.info('Building model...')
    model = Seq2Seq(word_vectors=word_vectors,
                    vocab_size=vocab_size,
                    hidden_size=args.hidden_size,
                    output_size=word_vectors.size(0),
                    drop_prob=args.drop_prob)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler    
    # Default project starter code uses Adadelta, but we're going to use Adam

    #optimizer = optim.Adadelta(model.parameters(), args.lr,
    #                           weight_decay=args.l2_wd)

    optimizer = optim.Adam(model.parameters(), args.lr,
                           weight_decay=args.l2_wd)
                               
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

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
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
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
                tgt_word_num_to_predict = torch.sum(q_len).item()
                loss_batch = F.nll_loss(log_p, qw_idxs_target, ignore_index=0, reduction='sum')
                loss = loss_batch / batch_size
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')

                    ema.assign(model)
                    results = evaluate(model,
                                       dev_loader,
                                       device,
                                       args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')


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
    with torch.no_grad(), \
        tqdm(total=len(data_loader.dataset)) as progress_bar:
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

            # Calculate perplexity        
            cum_loss += loss.item()
            q_mask = torch.zeros_like(qw_idxs_target) != qw_idxs_target
            q_len = q_mask.sum(-1)
            tgt_word_num_to_predict = torch.sum(q_len).item()
            cum_tgt_words += tgt_word_num_to_predict

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

        ppl = np.exp(cum_loss / cum_tgt_words)

    results_list = [('NLL', nll_meter.avg), \
                ('PPL', ppl)]
    results = OrderedDict(results_list)

    if was_training:
        model.train()

    return results


if __name__ == '__main__':
    main(get_train_args())
