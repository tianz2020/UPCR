import json

import ipdb
import torch
import torch.nn as nn
from tools import Tools
from option import option as op

class Action(nn.Module):
    def __init__(self,p_encoder,a_decoder,graphencoder,hidden_size,main_encoder,
                 m_encoder,n_topic_vocab,n_movie_vocab,bos_idx,max_len,glo2loc,loc2glo,vocab):
        super(Action, self).__init__()
        self.p_encoder = p_encoder
        self.a_decoder = a_decoder
        self.m_encoder = m_encoder
        self.main_encoder = main_encoder
        self.graphencoder = graphencoder
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.n_topic_vocab = n_topic_vocab
        self.n_movie_vocab = n_movie_vocab
        self.bos_idx = bos_idx
        self.max_len = max_len
        self.glo2loc = glo2loc
        self.loc2glo = loc2glo
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size,self.n_movie_vocab))

        self.a_linear = nn.Sequential(nn.Linear(hidden_size * 6, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, n_movie_vocab),
                                      nn.Softmax(-1))

    def forward(self,
                m,
                l,
                context,
                context_len,
                related_topics,related_topics_len,
                ar_gth,ar_gth_len,
                state,state_len,
                movie_path,movie_path_len,
                relations,relations_len,
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
            l = one_hot_scatter(l, self.n_topic_vocab)

        # preference
        m_mask = m.new_ones(m.size(0),1,m.size(1))
        m_hidden = self.p_encoder(m,m_mask)  # [B,L,H]

        # profile
        l_mask = l.new_ones(l.size(0),1,l.size(1))
        l_hidden = self.p_encoder(l,l_mask)  # [B,L,H]

        # movie path
        movie_mask = Tools.get_mask_via_len(movie_path_len,op.movie_path_len)
        movie_hidden = self.m_encoder(movie_path,movie_mask)

        # movie path topics
        mv_path_topic_hidden, mv_path_mask, mv_path_topic = get_movie_rep(movie_path, movie_mask, self.p_encoder, relations, relations_len, 3)
        mv_path_topic = one_hot_scatter(mv_path_topic, self.n_topic_vocab)

        # related movies
        related_topics_mask = Tools.get_mask_via_len(related_topics_len, op.movie_num)
        related_topic_hidden = self.m_encoder(related_topics,related_topics_mask)

        # context
        context_mask = Tools.get_mask_via_len(context_len, op.context_max_len)  # [B,1,L]
        context_hidden = self.main_encoder(context, context_mask)

        # union
        src_hidden = torch.cat([related_topic_hidden,movie_hidden,mv_path_topic_hidden,m_hidden,l_hidden,context_hidden],1)
        src_mask = torch.cat([related_topics_mask,movie_mask,mv_path_mask,m_mask,l_mask,context_mask],2)

        # init
        probs = None  # [B,1,V]
        action_mask = Tools.get_mask_via_len(ar_gth_len, op.action_num)

        if mode == 'train':
            # train
            for i in range(0,op.action_num,2):
                # src_rep = torch.cat([m_rep, l_rep, state_rep, context_rep, movie_rep, related_topic_rep], 2)
                # prob = self.a_linear(src_rep)
                seq_gth = ar_gth[:,0: i+1]
                ar_mask = action_mask[:,:,0:i+1]
                dec_output = Tools._single_decode(seq_gth.detach(), src_hidden, src_mask, self.a_decoder, ar_mask)
                prob = self.proj(dec_output, src_hidden, src_mask, m, l, related_topic_hidden, related_topics_mask,
                                 related_topics)

                if i == 0:
                    probs = prob
                else:
                    probs = torch.cat([probs, prob], 1)
            return probs
        else:
            # test
            seq_gen = None
            for i in range(0,op.action_num,2):
                # src_rep = torch.cat([m_rep, l_rep, state_rep, context_rep, movie_rep, related_topic_rep], 2)
                # single_step_prob = self.a_linear(src_rep)
                if i == 0:
                    seq_gen = ar_gth[:,0:i+1]
                else:
                    seq_gen = torch.cat([seq_gen,ar_gth[:,i:i+1]],1 )
                ar_mask = action_mask[:, :, 0:i + 1]
                dec_output = Tools._single_decode(seq_gen.detach(), src_hidden, src_mask, self.a_decoder,ar_mask)
                single_step_prob = self.proj(dec_output,src_hidden,src_mask,m,l,related_topic_hidden,
                                             related_topics_mask,related_topics)  # [B,1,V]

                # logit = self.gen_proj(dec_output)
                # single_step_prob = torch.softmax(logit, -1)
                # single_step_prob = self.gen_proj(dec_output)
                # single_step_prob = torch.softmax(single_step_prob, -1)
                if i == 0:
                    probs = single_step_prob
                else:
                    probs = torch.cat([probs, single_step_prob], 1)
                single_step_word = torch.argmax(single_step_prob, -1)  # [B,1]
                seq_gen = torch.cat([seq_gen,single_step_word],1)

            return seq_gen, probs  # [B,L]   [B,L,V]

    def proj(self,dec_out,src_hidden,src_mask,pv_m,l,related_topics_hidden,related_topics_mask,related_topics):
        '''
        生成概率 + 从src_hidden中复制的概率
        src_hidden = torch.cat([related_topic_hidden,movie_hidden,m_hidden,l_hidden,context_hidden],1)
        return : [B,L_a,V]
        '''
        L_a = dec_out.size(1)

        # generation  [B,L,V]
        gen_logit = self.gen_proj(dec_out)

        copy_hidden = src_hidden[:,0:op.preference_num + op.profile_num + 2 + op.state_num,:]
        copy_mask = src_mask[:,:,0:op.preference_num + op.profile_num + 2 + op.state_num ]

        # copy from context , m , l , state
        copy_logit = torch.bmm(dec_out, related_topics_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((related_topics_mask == 0).expand(-1, L_a, -1), -1e9)
        logits = torch.cat([gen_logit, copy_logit], -1)

        if op.scale_prj:
            logits *= self.hidden_size ** -0.5

        # logits -> probs
        probs = torch.softmax(logits, -1)
        # generation [B,L_a,V]
        gen_prob = probs[:, :, :self.n_movie_vocab]

        # # used for topic prediction
        # # copy from preference
        # copy_m_prob =probs[:,:,self.n_topic_vocab : self.n_topic_vocab + op.preference_num+1]
        # copy_m_prob = torch.bmm(copy_m_prob,pv_m) # [B,L,V]
        #
        # # copy from profile
        # copy_l_prob = probs[:,:,self.n_topic_vocab + op.preference_num +1 : self.n_topic_vocab + op.preference_num +1 + op.profile_num +1]
        # copy_l_prob = torch.bmm(copy_l_prob,l)  # [B,L,V]
        #
        # # copy_related_topics_prob
        # related_topics = one_hot_scatter(related_topics,self.n_topic_vocab)
        # copy_related_topics_prob = probs[:, :,
        #               self.n_topic_vocab + op.preference_num + op.profile_num + 2: ]
        # copy_related_topics_prob = torch.bmm(copy_related_topics_prob,related_topics)

        # used for recommendation
        related_topics = one_hot_scatter(related_topics, self.n_movie_vocab)
        copy_related_topics_prob = probs[:,:,self.n_movie_vocab:]
        copy_related_topics_prob = torch.bmm(copy_related_topics_prob, related_topics)

        probs = gen_prob + copy_related_topics_prob
        return probs


def get_movie_rep(related_movies,related_movies_mask,p_encoder,relations,relations_len,movie_num):
    '''
    related_movies : [B,L]  相关的电影数量
    related_movies_mask : [B,1,100]
    return: [B,L,H]  电影的表示
    '''

    '''
    related_movies : [B,L]  相关的电影数量
    related_movies_mask : [B,1,100]
    return: [B,L,H]  电影的表示
    '''
    movie_rep = None
    mask = None
    topics = None
    for i in range(movie_num):
        related_movie = related_movies[:,i]  # [B,]
        movie_mask = related_movies_mask[:,:,i]
        movie_topic_mask = movie_mask.unsqueeze(-1).expand(-1,-1,op.tag_num) # B,1,3
        related_topics = relations[related_movie,:op.tag_num]   # B , 3
        topics_len = relations_len[related_movie]
        topics_mask = Tools.get_mask_via_len(topics_len,op.tag_num)  # B,1,3
        topic_mask = movie_topic_mask & topics_mask  # B,1,3

        topic_hidden = p_encoder(related_topics,topic_mask)    # B,3,512
        # movie_hidden = torch.mean(topic_hidden,1).unsqueeze(1)  # B,1,512

        if movie_rep is None:
            movie_rep = topic_hidden
            mask = topic_mask
            topics = related_topics
        else:
            movie_rep = torch.cat([movie_rep,topic_hidden],1)
            mask = torch.cat([mask,topic_mask],2)
            topics = torch.cat([topics,related_topics],1)

    return movie_rep, mask, topics

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder