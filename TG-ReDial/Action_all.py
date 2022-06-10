import json
import torch.nn.functional as F
import ipdb
import torch
import torch.nn as nn
from tools import Tools
from option import option as op

class Action(nn.Module):
    def __init__(self,p_encoder,a_decoder,hidden_size,main_encoder,
                 m_encoder,n_topic_vocab,bos_idx,max_len,glo2loc,loc2glo,vocab):
        super(Action, self).__init__()
        self.p_encoder = p_encoder
        self.a_decoder = a_decoder
        self.m_encoder = m_encoder
        self.main_encoder = main_encoder
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.n_topic_vocab = n_topic_vocab
        self.bos_idx = bos_idx
        self.max_len = max_len
        self.glo2loc = glo2loc
        self.loc2glo = loc2glo
        self.gen_proj = nn.Linear(self.hidden_size,self.n_topic_vocab)
        self.topic2movie = nn.Linear(2583,self.n_topic_vocab-2583)
        self.mask = torch.zeros(op.batch_size,1,self.n_topic_vocab).cuda()
        self.mask[:,:,2583:] = 1
        self.pad = torch.zeros(op.batch_size,1,2583).cuda()
    def forward(self,m,l,context,context_len,ar_gth,ar_gth_len,
                tp_path,tp_path_len,related_movies,related_movies_len,mode):
        if mode == 'test':
            m = one_hot_scatter(m, self.n_topic_vocab)
            l = one_hot_scatter(l, self.n_topic_vocab)
        m_mask = m.new_ones(m.size(0),1,m.size(1))
        m_hidden = self.p_encoder(m,m_mask)
        l_mask = l.new_ones(l.size(0),1,l.size(1))
        l_hidden = self.p_encoder(l,l_mask)
        tp_path = one_hot_scatter(tp_path, self.n_topic_vocab)
        tp_mask = Tools.get_mask_via_len(tp_path_len, op.state_num)
        tp_path_hidden = self.p_encoder(tp_path, tp_mask)
        context_mask = Tools.get_mask_via_len(context_len, op.context_max_len)
        context_hidden = self.main_encoder(context, context_mask)
        related_movies = one_hot_scatter(related_movies,self.n_topic_vocab)
        related_movies_mask = Tools.get_mask_via_len(related_movies_len,op.movie_num)
        related_movies_hidden = self.p_encoder(related_movies,related_movies_mask)
        src_hidden = torch.cat([m_hidden,l_hidden,context_hidden,tp_path_hidden,related_movies_hidden],1)
        src_mask = torch.cat([m_mask,l_mask,context_mask,tp_mask,related_movies_mask],2)
        action_mask = Tools.get_mask_via_len(ar_gth_len,op.action_num)
        if mode == 'train':
            seq_gth = ar_gth[:,[0]]
            ar_mask = action_mask[:,:,[0]]
            dec_output = Tools._single_decode(seq_gth.detach(), src_hidden, src_mask, self.a_decoder, ar_mask)
            prob = self.proj(dec_out=dec_output, src_hidden=src_hidden, src_mask=src_mask,
                             tp=tp_path, m=m, l=l, context=context,related_movies=related_movies)
            return prob
        else:
            seq_gen = ar_gth[:,[0]]
            ar_mask = action_mask[:, :, [0]]
            dec_output = Tools._single_decode(seq_gen.detach(), src_hidden, src_mask, self.a_decoder,ar_mask)
            prob = self.proj(dec_out=dec_output, src_hidden=src_hidden, src_mask=src_mask,
                             tp=tp_path, m=m, l=l, context=context,related_movies=related_movies)
            word = torch.argmax(prob, -1)
            return word, prob

    def proj(self,dec_out, src_hidden,src_mask, tp, m, l, context,related_movies ):
        B,L_a =dec_out.size(0), dec_out.size(1)
        gen_logit = self.gen_proj(dec_out)
        copy_logit = torch.bmm(dec_out, src_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((src_mask == 0).expand(-1, L_a, -1), -1e9)
        logits = torch.cat([gen_logit, copy_logit], -1)
        if op.scale_prj:
            logits *= self.hidden_size ** -0.5
        probs = torch.softmax(logits, -1)
        gen_prob = probs[:, :, :self.n_topic_vocab]
        copy_m = probs[:, :, self.n_topic_vocab :
                             self.n_topic_vocab + op.preference_num]
        copy_m_prob = torch.bmm(copy_m, m)
        copy_l = probs[:, :, self.n_topic_vocab+ op.preference_num:self.n_topic_vocab+ op.preference_num+ op.profile_num]
        copy_l_prob = torch.bmm(copy_l, l)
        copy_context_prob = probs[:, :, self.n_topic_vocab+ op.preference_num+ op.profile_num:self.n_topic_vocab+ op.preference_num+ op.profile_num+ op.context_max_len]
        transfer_context_word = torch.gather(self.glo2loc.unsqueeze(0).expand(B, -1), 1, context)
        copy_context_temp = copy_context_prob.new_zeros(B, L_a, self.n_topic_vocab)
        copy_context_prob = copy_context_temp.scatter_add(dim=2,
                                                          index=transfer_context_word.unsqueeze(1).expand(-1, L_a, -1),
                                                          src=copy_context_prob)
        copy_tp = probs[:,:,self.n_topic_vocab+ op.preference_num+ op.profile_num+ op.context_max_len:
                            self.n_topic_vocab+ op.preference_num+ op.profile_num+ op.context_max_len + op.state_num]
        copy_tp_prob = torch.bmm(copy_tp, tp)
        copy_relation = probs[:,:,self.n_topic_vocab+ op.preference_num+ op.profile_num+ op.context_max_len + op.state_num:]
        copy_relation = torch.bmm(copy_relation,related_movies)
        probs = gen_prob + copy_m_prob + copy_l_prob +  copy_context_prob + copy_tp_prob + copy_relation
        probs = probs.mul(self.mask)
        norm = torch.sum(probs,-1)
        norm = norm.unsqueeze(1)
        probs/=norm
        return probs

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder