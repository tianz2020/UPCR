# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", type=int, default=5000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epoch", type=int, default=10000)
    cfg = parser.parse_args()
    return cfg

def one_hot(indice, num_classes):
    I = torch.eye(num_classes)
    T = I[indice]
    T.requires_grad = False
    return T

class GumbelSoftmax(nn.Module):
    def __init__(self, origin_version=True, rep_penalize=False, reoper=10):
        super(GumbelSoftmax, self).__init__()

        self.origin_version = origin_version
        self.eps = 1e-24
        self.step = 0
        self.rep_penalize = rep_penalize
        self.reoper = reoper

    def forward(self, inp, tau, normed):
        if normed:
            inp = torch.log(inp + self.eps)

        if not self.origin_version:
            gk = -torch.log(-torch.log(torch.rand(inp.shape)))
            out = torch.softmax((inp + gk) / tau, dim=-1)
        else:
            if self.rep_penalize:
                expand_inp = inp.unsqueeze(1).expand(-1, self.reoper, -1, -1)
                out = torch.nn.functional.gumbel_softmax(expand_inp, tau=tau)
                max_index = out.argmax(-1)
                max_index = max_index.reshape(max_index.size(0), -1)
                max_index = max_index.detach().cpu().tolist()
                def find_index(rand_value, prob_list):
                    ceil = np.cumsum(prob_list[:-1])
                    index = (rand_value > ceil).astype(np.long).sum()
                    return int(index)
                batch_selected_indexs = []
                for b in range(expand_inp.size(0)):
                    c = Counter()
                    c.update(max_index[b])
                    index2prob = dict([(x, 1 / y) for x, y in c.most_common()])
                    probs = [index2prob[i] for i in max_index[b]]
                    probs_sum = sum(probs)
                    normalized_probs = [x / probs_sum for x in probs]
                    indexs = [find_index(random.random(), normalized_probs) for _ in range(expand_inp.size(2))]
                    batch_selected_indexs.append(indexs)
                B, _, S, T = out.shape
                flat_out = out.reshape(-1, T)
                indexs = torch.tensor(batch_selected_indexs).reshape(-1)
                indexs = indexs + torch.arange(B).unsqueeze(1).expand(-1, self.reoper).reshape(
                    -1) * self.reoper * S
                flat_out = flat_out.index_select(0, indexs)
                out = flat_out.reshape(B, S, -1)
            else:
                out = torch.nn.functional.gumbel_softmax(inp, tau=tau)
        return out

class Argmax(nn.Module):
    def __init__(self):
        super(Argmax, self).__init__()
    def forward(self, inp):
        return torch.argmax(inp, dim=-1)

class GUMBEL(nn.Module):
    def __init__(self, sample_num, hidden_size, is_train=False, gumbel_act=True):
        super(GUMBEL, self).__init__()
        self.is_train = is_train
        self.gumbel_act = gumbel_act
        self.embedding_layer = nn.Linear(sample_num, hidden_size)
        self.pred_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, sample_num))
        self.train_act1 = nn.Softmax(dim=-1)
        self.train_act2 = GumbelSoftmax()
        self.test_act3 = Argmax()

    def get_act_fn(self):
        act_fn = self.test_act3 if not self.is_train else (self.train_act2 if self.gumbel_act else self.train_act1)
        return act_fn

    def forward(self, sample):
        sample = sample.cuda()
        sample_embedding = self.embedding_layer(sample)
        pred = self.pred_layer(sample_embedding)
        ret = self.get_act_fn()(pred)
        return ret