"""Test model on SQuAD

code adapted from:
    > https://github.com/chrischute/squad
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import json

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import Seq2SeqGru, Seq2Seq, Seq2SeqAttn, TransformerModel
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    #Load Word2Idx
    log.info('Loading word2Idx...')
    word2Idx = json.loads(open(args.word2idx_file).read())
    idx2Word = {v: k for (k,v) in word2Idx.items()}
    
    #vocab_size = len(word2Idx)
    vocab_size = word_vectors.size(0)
    print(f"Vocab size: {vocab_size}")

    def getWords(idxList):
        words = []
        for i in idxList:
            words.append(idx2Word[i])
        return words

    def create_new_model():
        if args.model_type == "seq2seq":
            return Seq2Seq(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    output_size=vocab_size,
                    device=device,
                    num_layers=args.num_layers)
        elif args.model_type == "seq2seq_attn":
            return Seq2SeqAttn(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    output_size=vocab_size,
                    device=device,
                    num_layers=args.num_layers)
        elif args.model_type == "transformer":
            return TransformerModel(vocab_size, device)

    # Get model
    log.info('Building model...')
    model = create_new_model()
    model = nn.DataParallel(model, gpu_ids)

    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions by Beam Search
    eval_file = vars(args)[f'{args.split}_eval_file']
    candidates_corpus = []
    references_corpus = []
    context_file = os.path.join(args.save_dir, args.context_file)
    question_file = os.path.join(args.save_dir, args.question_file)
    prediction_file = os.path.join(args.save_dir, args.prediction_file)


    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        
        for cw_idxs, re_cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            re_cw_idxs = re_cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            cw_list = []
            qw_list = []

            # Forward
            for re_cw_idx, qw_idx in zip(torch.split(re_cw_idxs, split_size_or_sections=1, dim=0), torch.split(qw_idxs, split_size_or_sections=1, dim=0)):

                cw_list.append(re_cw_idx)
                qw_list.append(qw_idx)
                
                #y = F.one_hot(qw_idx, num_classes=len(word_vectors))
                #print(getWords(cw_idx.squeeze().tolist()))
                #print(getWords(qw_idx.squeeze().tolist()))
                #util.TeacherForce(model, word2Idx, Idx2Word, cw_idx, qw_idx, device)
                #util.evaluateRandomly(model, word2Idx, Idx2Word, cw_idx, device)

                '''

                hypotheses = util.beamSearch(model, word2Idx, Idx2Word, cw_idx, device)
                loss = 0.
                pred_dict[cw_idx] = []

                for hyp in hypotheses:
                    loss = loss + hyp.score
                    pred_dict[cw_idx].append(hyp.value)
                nll_meter.update(loss, batch_size)
                #wait = input("Sab chill hai.. press to continue")

                cw_list.append(cw_idx)
                qw_list.append(qw_idx)            
                '''
            candidates, references = util.find_candidates_and_references_for_bleu(model, word2Idx, idx2Word, 
                                                                                      cw_list, qw_list, device,
                                                                                      context_file, question_file, prediction_file)
            candidates_corpus += candidates
            references_corpus += references            
            progress_bar.update(batch_size)    

            # Log info
            #if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
            #    progress_bar.set_postfix(NLL=nll_meter.avg)
        print("Calculating corpus BLEU...")
        util.compute_corpus_level_bleu_score(model, args.split, candidates_corpus, references_corpus, device)
        
'''

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)
'''

if __name__ == '__main__':
    main(get_test_args())