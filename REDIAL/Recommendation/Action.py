import json
import torch.nn.functional as F
import ipdb
import random
import torch
import torch.nn as nn
from tools import Tools
from option import option as op

class Action(nn.Module):
    def __init__(self,p_encoder,a_decoder,graphencoder,hidden_size,main_encoder,
                 n_topic_vocab,bos_idx,max_len,glo2loc,loc2glo,vocab):
        super(Action, self).__init__()
        self.p_encoder = p_encoder
        self.a_decoder = a_decoder
        self.main_encoder = main_encoder
        self.graphencoder = graphencoder
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.n_topic_vocab = n_topic_vocab
        self.bos_idx = bos_idx
        self.max_len = max_len
        self.glo2loc = glo2loc
        self.loc2glo = loc2glo
        self.output_en = nn.Linear(self.hidden_size, self.n_topic_vocab)
        self.gen_proj = nn.Linear(self.hidden_size,self.n_topic_vocab)
        self.topic2movie = nn.Linear(2583,self.n_topic_vocab-2583)
        # self.mask = torch.zeros(op.batch_size, 1, self.n_topic_vocab).cuda()
        # self.movies = []
        # movie_id = json.load(open('./dataset/movie_id.json'))
        # for movie in movie_id:
        #     self.movies.append(int(movie))
        # self.movie_len = len(self.movies)

    def forward(self,
                m,
                l,
                context,
                context_len,
                ar_gth,ar_gth_len,
                tp_path,tp_path_len,
                related_topic,related_topic_len,
                tp_path_hidden,related_topic_hidden,
                mode):
        '''
        m :                 [B,L_m,V]
        l:                  [B,L_l,V]
        related topics :    [B,L]
        related_topics_len: [B,]
        turn:               [B,1]
        '''


        if mode == 'test':
            m = one_hot_scatter(m, self.n_topic_vocab)
            # l = one_hot_scatter(l, self.n_topic_vocab)
        bs = m.size(0)

        # preference
        m_mask = m.new_ones(m.size(0),1,m.size(1))
        m_hidden = self.p_encoder(m,m_mask)  # [B,L,H]

        # profile
        # l_mask = l.new_ones(l.size(0),1,l.size(1))
        # l_hidden = self.p_encoder(l,l_mask)  # [B,L,H]


        # topic path
        tp_path = one_hot_scatter(tp_path, self.n_topic_vocab)
        tp_mask = Tools.get_mask_via_len(tp_path_len, op.state_num_redial)

        # context
        context_mask = Tools.get_mask_via_len(context_len, op.context_max_len)  # [B,1,L]
        context_hidden = self.main_encoder(context, context_mask)

        # related_movies
        related_topic_mask = Tools.get_mask_via_len(related_topic_len,op.movie_num)
        related_topic = one_hot_scatter(related_topic,self.n_topic_vocab)

        # union
        src_hidden = torch.cat([m_hidden
                                   # ,l_hidden
                                   ,context_hidden,tp_path_hidden,related_topic_hidden],1)
        src_mask = torch.cat([m_mask,
                              # l_mask,
                              context_mask,tp_mask,related_topic_mask],2)
        seq_gen_at = Tools._generate_init(bs, self.n_topic_vocab, trg_bos_idx=self.bos_idx)
        if mode == 'train':
            # train
            dec_output = Tools._single_decode(seq_gen_at.detach(), src_hidden, src_mask, self.a_decoder)
            prob = self.proj(dec_out=dec_output, src_hidden=src_hidden, src_mask=src_mask,
                             tp=tp_path, m=m, l=l, context=context,related_movies=related_topic,
                             ar_gth=ar_gth)

            return prob

        else:
            dec_output = Tools._single_decode(seq_gen_at.detach(), src_hidden, src_mask, self.a_decoder)
            prob = self.proj(dec_out=dec_output, src_hidden=src_hidden, src_mask=src_mask,
                             tp=tp_path, m=m, l=l, context=context,related_movies=related_topic,
                             ar_gth=ar_gth)
            word = torch.argmax(prob, -1)  # [B,1]
            return word, prob  # [B,L]   [B,L,V]

    def proj(self,dec_out, src_hidden,src_mask, tp, m, l, context,related_movies,ar_gth ):
        '''
        src_hidden = torch.cat([m_hidden,l_hidden,context_hidden,tp_path_hidden],1)
        src_hidden = torch.cat([m_hidden,l_hidden,context_hidden,tp_path_hidden,related_topic_hidden],1)
        '''
        B,L_a =dec_out.size(0), dec_out.size(1)

        # generation  [B,L,V]
        gen_logit = self.gen_proj(dec_out)
        # prob = None


        copy_logit = torch.bmm(dec_out, src_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((src_mask == 0).expand(-1, L_a, -1), -1e9)

        logits = torch.cat([gen_logit, copy_logit], -1)

        if op.scale_prj:
            logits *= self.hidden_size ** -0.5

        # logits -> probs
        probs = torch.softmax(logits, -1)

        # generation [B,L_a,V]
        gen_prob = probs[:, :, :self.n_topic_vocab]

        # preference
        copy_m = probs[:, :, self.n_topic_vocab :
                              self.n_topic_vocab + op.preference_num]
        copy_m_prob = torch.bmm(copy_m, m)

        # profile
        # copy_l = probs[:, :, self.n_topic_vocab + op.preference_num:
        #                      self.n_topic_vocab + op.preference_num + op.profile_num]
        # copy_l_prob = torch.bmm(copy_l, l)

        # context
        copy_context_prob = probs[:, :, self.n_topic_vocab + op.preference_num
                                        # + op.profile_num
                                        :
                                        self.n_topic_vocab + op.preference_num
                                        # + op.profile_num
                                        + op.context_max_len]
        transfer_context_word = torch.gather(self.glo2loc.unsqueeze(0).expand(B, -1), 1, context)  # glo_idx to loc_idx
        copy_context_temp = copy_context_prob.new_zeros(B, L_a, self.n_topic_vocab)
        copy_context_prob = copy_context_temp.scatter_add(dim=2,
                                                          index=transfer_context_word.unsqueeze(1).expand(-1, L_a, -1),
                                                          src=copy_context_prob)

        # topic path
        copy_tp = probs[:,:,self.n_topic_vocab + op.preference_num
                            # + op.profile_num
                            + op.context_max_len:
                            self.n_topic_vocab + op.preference_num
                            # + op.profile_num
                            + op.context_max_len + op.state_num_redial]
        copy_tp_prob = torch.bmm(copy_tp, tp)

        # related_movie
        copy_relation = probs[:,:,self.n_topic_vocab + op.preference_num
                                  # + op.profile_num
                                  + op.context_max_len + op.state_num_redial:]
        copy_relation = torch.bmm(copy_relation,related_movies)

        probs = gen_prob + copy_m_prob + copy_context_prob + copy_tp_prob + copy_relation
                # + copy_l_prob\

        # probs = probs.mul(self.mask)
        # norm = torch.sum(probs,-1)
        # norm = norm.unsqueeze(1)
        # probs/=norm

        return probs


def get_movie_rep(related_movies,related_movies_mask,relations,relations_len,p_encoder,movie_num):
    '''
    related_movies : [B,L]  相关的电影数量
    related_movies_mask : [B,1,100]
    return: [B,L,H]  电影的表示
    '''
    mask = None
    topics = None
    masks = None
    for i in range(movie_num):
        related_movie = related_movies[:,i]  # [B,]
        movie_mask = related_movies_mask[:,:,i]
        movie_topic_mask = movie_mask.unsqueeze(-1).expand(-1,-1,op.tag_num) # B,1,3
        related_topics = relations[related_movie,:op.tag_num]   # B , 3
        topics_len = relations_len[related_movie]
        topics_mask = Tools.get_mask_via_len(topics_len,op.tag_num)  # B,1,3
        topic_mask = movie_topic_mask & topics_mask  # B,1,3

        related_movie = related_movie.unsqueeze(-1)
        movie_mask = movie_mask.unsqueeze(-1)

        mask = torch.cat([movie_mask, topic_mask], 2)

        if topics is None:
            topics = torch.cat([related_movie, related_topics], 1)
            masks = mask
        else:
            topics = torch.cat([topics, related_movie], 1)
            topics = torch.cat([topics, related_topics], 1)
            masks = torch.cat([masks,mask],2)

    return topics, masks

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder