import os
import math

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from build_vocab import PAD_TOKEN, UNK_TOKEN


def collate_fn(sign2id, batch):
    """
    Modifies and merges a list of samples appropriately to form a mini-batch of
    Tensor(s).

    Inputs:
    - sign2id: (dict) Vocabulary mapping from sign to id
    - batch: list of input tuples. Each input tuple is of the format
        (formula_img, coordinates, symbols, edge_indices, formula)
      - formula_img [1, Hi, Wi]
      - coordinates [4, Li]
      - symbols [Li, 1, 32, 32]
      - edge_indices [2, 2E]
      - formula (string)

    Outputs: tuple of the following items
    - formula_img: (torch.Tensor) [N, 1, H, W]
    - coordinates: (torch.Tensor) [N, 4, L]
    - symbols: (torch.Tensor) [N, L, 1, 32, 32]
    - edge_indices: (torch.Tensor) [N, 2, 2E]
    - seq_lens: (torch.Tensor) [N]
    - tgt4training: (torch.Tensor) [N, T+1]
      list in the batch
    - tgt4cal_loss: (torch.Tensor) [N, T+1]

      Here,
        N -> Batch size
        H -> Height of formula image
        W -> Width of formula image
        L -> Number of symbols in the image with the largest number of symbols
        E -> Number of edges in the undirected graph
        T -> size of longest token list in the batch
    """
    ## filter the pictures that have different weight or height
    #size = batch[0][0].size()
    #batch = [img_formula for img_formula in batch
    #         if img_formula[0].size() == size]
    seq_lens = [coord.shape[1] for _, coord, _, _, _ in batch]
    ## pad different sized batches appropriately
    max_h = max([img.shape[1] for img, _, _, _, _ in batch])
    max_w = max([img.shape[2] for img, _, _, _, _ in batch])
    max_l = max(seq_lens)
    for i in range(len(batch)):
        f_img = batch[i][0] #formula img
        coord = batch[i][1]
        f_img = F.pad(f_img, (0, max_w-f_img.shape[2], 0, max_h-f_img.shape[1]),
                          mode="constant", value=1.0) #pad with white pixels
        coord = F.pad(coord, (0, max_l-coord.shape[1]),
                          mode="constant", value=0.0) #pad with white pixels
        batch[i] = (f_img, coord, batch[i][2], batch[i][3], batch[i][4])
    # Modify edges accordingly i.e. diagonalize batch adjacency matrix
    idx_shift = 0
    for i in range(len(batch)):
        edge_index = batch[i][3] + idx_shift
        batch[i] = (batch[i][0], batch[i][1], batch[i][2], edge_index, batch[i][4])
        idx_shift += seq_lens[i]

    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula[-1].split()),
               reverse=True)

    formula_imgs, coordinates, symbols, edge_indices, formulas = zip(*batch)
    formulas = [formula.split() for formula in formulas]
    # targets for training , begin with START_TOKEN
    tgt4training = formulas2tensor(add_start_token(formulas), sign2id)
    # targets for calculating loss , end with END_TOKEN
    tgt4cal_loss = formulas2tensor(add_end_token(formulas), sign2id)
    formula_imgs = torch.stack(formula_imgs, dim=0).float()         # [N, 1, H, W]
    coordinates = torch.stack(coordinates, dim=0).float()          # [N, 4, L] where L = max(seq_lens)
    symbols = torch.cat(symbols, dim=0).float()                     # [L', 1, 32, 32] where L' = sum(seq_lens)
    edge_indices = torch.concat(edge_indices, dim=1).long()        # [2, 2E'] where E' is the num of edges in all graphs in the batch
    seq_lens = torch.tensor(seq_lens, dtype=torch.int64)     # [N]
    return formula_imgs, coordinates, symbols, edge_indices, seq_lens, tgt4training, tgt4cal_loss


def formulas2tensor(formulas, sign2id):
    """convert formula to tensor"""

    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors


def add_start_token(formulas):
    return [['<s>']+formula for formula in formulas]


def add_end_token(formulas):
    return [formula+['</s>'] for formula in formulas]


def count_parameters(model):
    """count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()
    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas


def cal_loss(logits, targets):
    """args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """
    padding = torch.ones_like(targets) * PAD_TOKEN
    mask = (targets != padding)

    targets = targets.masked_select(mask)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, logits.size(2))
    ).contiguous().view(-1, logits.size(2))
    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)

    loss = F.nll_loss(logits, targets)
    return loss


def get_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt')]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        epoch = int(ckpt.split('-')[1])
        if epoch > max_epoch:
            max_epoch = epoch
            last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path


def schedule_sample(prev_logit, prev_tgt, epsilon):
    prev_out = torch.argmax(prev_logit, dim=1, keepdim=True)
    prev_choices = torch.cat([prev_out, prev_tgt], dim=1)  # [B, 2]
    batch_size = prev_choices.size(0)
    prob = Bernoulli(torch.tensor([epsilon]*batch_size).unsqueeze(1))
    # sampling
    sample = prob.sample().long().to(prev_tgt.device)
    next_inp = torch.gather(prev_choices, 1, sample)
    return next_inp


def cal_epsilon(k, step, method):
    """
    Reference:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        See details in https://arxiv.org/pdf/1506.03099.pdf
    """
    assert method in ['inv_sigmoid', 'exp', 'teacher_forcing']

    if method == 'exp':
        return k**step
    elif method == 'inv_sigmoid':
        return k/(k+math.exp(step/k))
    else:
        return 1.
