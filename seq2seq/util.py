"""Utility classes and methods.

code adapted from:
    > https://github.com/chrischute/squad
"""
import logging
import os
import queue
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import ujson as json

from collections import Counter
from collections import namedtuple
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
from typing import List, Tuple, Dict, Set, Union
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

SOS = "--SOS--"
EOS = "--EOS--"
NULL = "--NULL--"

class SQuAD(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).

    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """
    def __init__(self, data_path, use_v2=True):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.reduced_context_idxs = torch.from_numpy(dataset['reduced_context_idxs'])
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()

        if use_v2:
            # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
            batch_size, c_len, w_len = self.context_char_idxs.size()
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or 'test' in data_path or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.reduced_context_idxs[idx], 
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.valid_idxs)


def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `SQuAD.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, y1s, y2s, ids).

    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs, question_idxs,
        question_char_idxs, y1s, y2s, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.

    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    context_idxs, reduced_context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        y1s, y2s, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    reduced_context_idxs = merge_1d(reduced_context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)

    return (context_idxs, reduced_context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        best_model_name (str): File name of the best model.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, best_model_name, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.best_model_name = best_model_name
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, self.best_model_name)
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def visualize(tbx, pred_dict, eval_path, step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

    with open(eval_path, 'r') as eval_file:
        eval_dict = json.load(eval_file)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        example = eval_dict[str(id_)]
        question = example['question']
        context = example['context']
        answers = example['answers']

        gold = answers[0] if answers else 'N/A'
        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag=f'{split}/{i+1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)


def save_preds(preds, save_dir, file_name='predictions.csv'):
    """Save predictions `preds` to a CSV file named `file_name` in `save_dir`.

    Args:
        preds (list): List of predictions each of the form (id, start, end),
            where id is an example ID, and start/end are indices in the context.
        save_dir (str): Directory in which to save the predictions file.
        file_name (str): File name for the CSV file.

    Returns:
        save_path (str): Path where CSV file was saved.
    """
    # Validate format
    if (not isinstance(preds, list)
            or any(not isinstance(p, tuple) or len(p) != 3 for p in preds)):
        raise ValueError('preds must be a list of tuples (id, start, end)')

    # Make sure predictions are sorted by ID
    preds = sorted(preds, key=lambda p: p[0])

    # Save to a CSV file
    save_path = os.path.join(save_dir, file_name)
    np.savetxt(save_path, np.array(preds), delimiter=',', fmt='%d')

    return save_path


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def teacher_force(model, word2idx_dict, idx2word_dict, cw_idx, qw_idx, device):
    max_len = 2

    q = qw_idx.squeeze().tolist()
    _, dec_init_state = model.module.encode(cw_idx)
    
    prev_word = SOS
    h_t = dec_init_state

    t = 0
    sent = []

    while prev_word != EOS and t < max_len:
        sent.append(prev_word)
        eos_id = word2idx_dict[EOS] 
        y_tm1 = torch.tensor([[q[t]]], dtype=torch.long, device=device)

        h_t, log_p_t  = model.module.decode(h_t, y_tm1)
                
        prev_word = idx2word_dict[log_p_t.argmax(-1).squeeze().tolist()]
        t = t+1

    print(sent)

def greedy_decode(model, word2idx_dict, idx2word_dict, cw_idx, device):
    max_len = 10
    _, dec_init_state = model.module.encode(cw_idx)
    
    prev_word = SOS
    h_t = dec_init_state

    t = 0
    sent = []
    pred_idxs = []

    while prev_word != EOS and t < max_len:
        sent.append(prev_word)
        eos_id = word2idx_dict[EOS] 
        y_tm1 = torch.tensor([[word2idx_dict[prev_word]]], dtype=torch.long, device=device)
        h_t, log_p_t  = model.module.decode(h_t, y_tm1)
        pred_idx = log_p_t.argmax(-1).squeeze().item()
        pred_idxs.append(pred_idx)
        #prev_word = idx2word_dict[log_p_t.argmax(-1).squeeze().tolist()]
        prev_word = idx2word_dict[pred_idx]
        t = t+1
    
    return pred_idxs

def evaluate(model, word2idx_dict, idx2word_dict, cw_idx, device):
    max_len = 30
     _, dec_init_state = model.module.encode(cw_idx)
    
    prev_word = SOS
    h_t = dec_init_state

    t = 0
    sent = []

    while prev_word != EOS and t < max_len:
        sent.append(prev_word)
        eos_id = word2idx_dict[EOS] 
        y_tm1 = torch.tensor([[word2idx_dict[prev_word]]], dtype=torch.long, device=device)        
        h_t, log_p_t  = model.module.decode(h_t, y_tm1)        
        prev_word = idx2word_dict[log_p_t.argmax(-1).squeeze().tolist()]
        t = t+1

    return sent

def get_top_hypothesis(model, word2idx_dict, idx2word_dict,
                      cw_idx, qw_idx, device,
                      context_file, gold_question_file, prediction_file):
    #output_words = evaluate(model, word2idx_dict, idx2word_dict, cw_idx, device)
    #output_sentence = ' '.join(output_words)
    
    if model.module.model_type == 'seq2seq':
        hypotheses = beam_search(model, cw_idx, word2idx_dict, idx2word_dict, device, beam_size=3, max_decoding_time_step=20)
    elif model.module.model_type == 'seq2seq_attn':
        hypotheses = beam_search_attention(model, cw_idx, word2idx_dict, idx2word_dict, device, beam_size=3, max_decoding_time_step=20)

    top_hypothesis = hypotheses[0]
    gold_question = ' '.join([idx2word_dict[id] for id in qw_idx.squeeze().tolist()])
    context = ' '.join([idx2word_dict[id] for id in cw_idx.squeeze().tolist()])
    
    context = context.replace(SOS, "").replace(EOS, "").replace(NULL, "")
    gold_question = gold_question.replace(SOS, "").replace(EOS, "").replace(NULL, "")
    hyp_question = ' '.join(top_hypothesis.value)

    if os.path.exists(context_file):
        mode = 'a'
    else:
        mode = 'w'
    
    with open(context_file, mode, encoding='utf-8') as cf, \
                open(gold_question_file, mode, encoding='utf-8') as qf, \
                    open(prediction_file, mode, encoding='utf-8') as pf:
                        cf.write(context + '\n')
                        qf.write(gold_question + '\n')
                        pf.write(hyp_question + '\n')
    
    return context, gold_question, top_hypothesis


def find_candidates_and_references_for_bleu(model, word2idx_dict, idx2word_dict,
                                            cw_idx_list, qw_idx_list, device,
                                            context_file, gold_question_file, prediction_file):
    references  = {}
    candidates = {}
    for x,y in list(zip(cw_idx_list, qw_idx_list)):
        c, q, h = get_top_hypothesis(model, word2idx_dict, idx2word_dict,
                                     x, y, device, context_file,
                                     gold_question_file, prediction_file)
        if c in references.keys():
            references[c].append(q.split())
        else:
            references[c] = [q.split()]
        candidates[c] = h.value
    return list(candidates.values()), list(references.values())


def compute_corpus_level_bleu_score(model, set_name, candidates_corpus, references_corpus, device):
    for i in range(1,5):        
        bleu_torch = bleu_score(candidates_corpus, references_corpus, max_n=i, weights=[1./i]*i)
        print("BLEU-" + str(i) + " on " + set_name + " using torchtext :" + str(bleu_torch))
        bleu_nltk = corpus_bleu(references_corpus, candidates_corpus, weights=[1./i]*i)
        print("BLEU-" + str(i) + " on " + set_name + " using nltk :" + str(bleu_nltk))


def beam_search(model, cw_idx, word2idx_dict, idx2word_dict, device, beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
    src_encodings, dec_init_vec = model.module.encode(cw_idx)
    h_tm1 = dec_init_vec
    
    eos_id = word2idx_dict[EOS]
    vocab_size = model.module.word_vectors.size(0)

    hypotheses = [[SOS]]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)

        y_tm1 = torch.tensor([[word2idx_dict[hyp[-1]]] for hyp in hypotheses], dtype=torch.long, device=device)
        
        (h_t, cell_t), log_p_t = model.module.decode(h_tm1, y_tm1)
        h_t = h_t.transpose(0,1)
        cell_t = cell_t.transpose(0,1)
        
        live_hyp_num = beam_size - len(completed_hypotheses)
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).unsqueeze(2).expand_as(log_p_t) + log_p_t).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

        prev_hyp_ids = top_cand_hyp_pos / vocab_size
        hyp_word_ids = top_cand_hyp_pos % vocab_size

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = idx2word_dict[hyp_word_id]
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
            if hyp_word == EOS:
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                        score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)
        
        h_t, cell_t = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
        h_t = h_t.transpose(0,1)
        cell_t = cell_t.transpose(0,1)
        h_tm1 = (h_t, cell_t)
        
        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses


def beam_search_attention(model, cw_idx, word2idx_dict, idx2word_dict, device, beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        
    src_encodings, dec_init_vec = model.module.encode(cw_idx)
    src_encodings_att_linear = model.module.att_projection(src_encodings)

    h_tm1 = dec_init_vec
    att_tm1 = torch.zeros(1, model.module.hidden_size, device=device)
    att_tm1 = att_tm1.unsqueeze(1)

    eos_id = word2idx_dict[EOS]
    vocab_size = model.module.word_vectors.size(0)

    hypotheses = [[SOS]]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)

        exp_src_encodings = src_encodings.expand(hyp_num,
                                                 src_encodings.size(1),
                                                 src_encodings.size(2))

        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                       src_encodings_att_linear.size(1),
                                                                       src_encodings_att_linear.size(2))

        y_tm1 = torch.tensor([[word2idx_dict[hyp[-1]]] for hyp in hypotheses], dtype=torch.long, device=device)
        y_t_embed = model.module.emb(y_tm1)
        
        x = torch.cat([y_t_embed, att_tm1], dim=-1)

        (h_t, cell_t), att_t, _  = model.module.step(x, h_tm1,
                                                  exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

        h_t = h_t.transpose(0,1)
        cell_t = cell_t.transpose(0,1)
        
        log_p_t = model.module.generator(att_t)

        live_hyp_num = beam_size - len(completed_hypotheses)
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).unsqueeze(2).expand_as(log_p_t) + log_p_t).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

        prev_hyp_ids = top_cand_hyp_pos / vocab_size
        hyp_word_ids = top_cand_hyp_pos % vocab_size

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = idx2word_dict[hyp_word_id]
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
            if hyp_word == EOS:
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                        score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)
        
        h_t, cell_t = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
        h_t = h_t.transpose(0,1)
        cell_t = cell_t.transpose(0,1)
        h_tm1 = (h_t, cell_t)
        att_tm1 = att_t[live_hyp_ids]

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses


def convert_tokens(eval_dict, qa_id, y_start_list, y_end_list, no_answer):
    """Convert predictions to tokens from the context.

    Args:
        eval_dict (dict): Dictionary with eval info for the dataset. This is
            used to perform the mapping from IDs and indices to actual text.
        qa_id (int): List of QA example IDs.
        y_start_list (list): List of start predictions.
        y_end_list (list): List of end predictions.
        no_answer (bool): Questions can have no answer. E.g., SQuAD 2.0.

    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted answer text.
        sub_dict (dict): Dictionary UUIDs -> predicted answer text (submission).
    """
    pred_dict = {}
    sub_dict = {}
    for qid, y_start, y_end in zip(qa_id, y_start_list, y_end_list):
        context = eval_dict[str(qid)]["context"]
        spans = eval_dict[str(qid)]["spans"]
        uuid = eval_dict[str(qid)]["uuid"]
        if no_answer and (y_start == 0 or y_end == 0):
            pred_dict[str(qid)] = ''
            sub_dict[uuid] = ''
        else:
            if no_answer:
                y_start, y_end = y_start - 1, y_end - 1
            start_idx = spans[y_start][0]
            end_idx = spans[y_end][1]
            pred_dict[str(qid)] = context[start_idx: end_idx]
            sub_dict[uuid] = context[start_idx: end_idx]
    return pred_dict, sub_dict


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_dicts(gold_dict, pred_dict, no_answer):
    avna = f1 = em = total = 0
    for key, value in pred_dict.items():
        total += 1
        ground_truths = gold_dict[key]['answers']
        prediction = value
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
        if no_answer:
            avna += compute_avna(prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    if no_answer:
        eval_dict['AvNA'] = 100. * avna / total

    return eval_dict


def compute_avna(prediction, ground_truths):
    """Compute answer vs. no-answer accuracy."""
    return float(bool(prediction) == bool(ground_truths))


# All methods below this line are from the official SQuAD 2.0 eval script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
