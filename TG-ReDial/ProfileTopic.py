from gumbel_softmax import GumbelSoftmax
from tau_scheduler import TauScheduler
import torch.nn as nn
import torch
from option import option as op
from tools import Tools

class PriorProfile(nn.Module):
    def __init__(self,encoder, decoder, hidden_size, n_topic_vocab,
                 trg_bos_idx, max_seq_len, gs: GumbelSoftmax, ts: TauScheduler):
        super(PriorProfile, self).__init__()
        self.id_encoder = encoder
        self.decoder = decoder
        self.n_topic_vocab = n_topic_vocab
        self.bos_idx = trg_bos_idx
        self.hidden_size = hidden_size
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size, self.n_topic_vocab))
        self.max_seq_len = max_seq_len
        self.gs = gs
        self.ts = ts

    def forward(self,id):
        bs = id.size(0)
        id_mask = id.new_ones(bs, 1, 1).cuda()
        user_id = id.unsqueeze(-1)
        id_hidden_p = self.id_encoder(user_id, id_mask)
        seq_gen_gumbel = Tools._generate_init(bs, self.n_topic_vocab, trg_bos_idx=self.bos_idx,training=self.training)
        seq_gen_prob = None
        for _ in range(op.profile_num):
            dec_output = Tools._single_decode(seq_gen_gumbel.detach(), id_hidden_p, id_mask, self.decoder)
            single_step_prob = self.gen_proj(dec_output)
            single_step_prob = torch.softmax(single_step_prob,-1)
            single_step_gumbel_word = self.gs.forward(single_step_prob, self.ts.step_on(), normed=True)
            if self.training:
                if seq_gen_prob is not None:
                    seq_gen_prob = torch.cat([seq_gen_prob, single_step_prob], 1)
                else:
                    seq_gen_prob = single_step_prob
                seq_gen_gumbel = torch.cat([seq_gen_gumbel, single_step_gumbel_word], 1)
            else:
                single_step_word = torch.argmax(single_step_prob, -1)
                seq_gen_gumbel = torch.cat([seq_gen_gumbel, single_step_word], 1)
        if self.training:
            return seq_gen_prob, seq_gen_gumbel[:,1:,:]
        else:
            return seq_gen_gumbel[:,1:]

class PosteriorProfile(nn.Module):
    def __init__(self,main_encoder,topic_encoder,id_encoder,decoder, hidden_size, n_topic_vocab,glo2loc,loc2glo,
                 trg_bos_idx, max_seq_len, gs: GumbelSoftmax, ts: TauScheduler):
        super(PosteriorProfile, self).__init__()
        self.main_encoder = main_encoder
        self.topic_encoder = topic_encoder
        self.id_encoder = id_encoder
        self.n_topic_vocab = n_topic_vocab
        self.bos_idx = trg_bos_idx
        self.glo2loc = glo2loc
        self.loc2glo = loc2glo
        self.hidden_size = hidden_size
        self.decoder = decoder
        self.max_seq_len = max_seq_len
        self.gs = gs
        self.ts = ts
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size, self.n_topic_vocab))

    def forward(self,id,topics,topics_len):
        bs = id.size(0)
        topics = one_hot_scatter(topics,self.n_topic_vocab)
        topic_mask = Tools.get_mask_via_len(topics_len, op.all_topic_num)
        topic_hidden = self.topic_encoder(topics,topic_mask)
        id_mask = id.new_ones(bs, 1, 1).cuda()
        user_id = id.unsqueeze(-1)
        id_hidden_q = self.id_encoder(user_id, id_mask)
        src_hidden = torch.cat([id_hidden_q,topic_hidden],1)
        src_mask = torch.cat([id_mask,topic_mask],2)
        seq_gen_gumbel = Tools._generate_init(bs, self.n_topic_vocab, trg_bos_idx=self.bos_idx,training=self.training)
        seq_gen_prob = None
        for _ in range(op.profile_num):
            dec_output = Tools._single_decode(seq_gen_gumbel.detach(), src_hidden, src_mask, self.decoder)
            single_step_prob = self.gen_proj(dec_output)
            single_step_prob = torch.softmax(single_step_prob, -1)
            if self.training:
                single_step_gumbel_word = self.gs.forward(single_step_prob, self.ts.step_on(),normed=True)
                if seq_gen_prob is not None:
                    seq_gen_prob = torch.cat([seq_gen_prob, single_step_prob], 1)
                else:
                    seq_gen_prob = single_step_prob
                seq_gen_gumbel = torch.cat([seq_gen_gumbel, single_step_gumbel_word], 1)
            else:
                single_step_word = torch.argmax(single_step_prob, -1)
                seq_gen_gumbel = torch.cat([seq_gen_gumbel, single_step_word], 1)
        if self.training:
            return seq_gen_prob, seq_gen_gumbel[:,1:,:]
        else:
            return seq_gen_gumbel[:,1:]

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder
