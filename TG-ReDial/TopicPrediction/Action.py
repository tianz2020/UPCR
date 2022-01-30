import json

import ipdb
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
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size,self.n_topic_vocab))
        

    def forward(self,
                m,
                l,
                context,
                context_len,
                related_topics,related_topics_len,
                ar_gth,ar_gth_len,
                tp_path,tp_path_len,
                mode,
                context_hidden=None):
      

        if mode == 'test':
            m = one_hot_scatter(m, self.n_topic_vocab)
            l = one_hot_scatter(l, self.n_topic_vocab)

        # preference
        m_mask = m.new_ones(m.size(0),1,m.size(1))
        m_hidden = self.p_encoder(m,m_mask) 

        # profile
        l_mask = l.new_ones(l.size(0),1,l.size(1))
        l_hidden = self.p_encoder(l,l_mask)  

        # related topics
        related_topics = one_hot_scatter(related_topics,self.n_topic_vocab)
        related_topics_mask = Tools.get_mask_via_len(related_topics_len, op.relation_num)
        related_topic_hidden = self.p_encoder(related_topics,related_topics_mask)


        # topic path
        tp_path = one_hot_scatter(tp_path, self.n_topic_vocab)
        tp_mask = Tools.get_mask_via_len(tp_path_len, op.state_num)
        tp_hidden = self.p_encoder(tp_path, tp_mask)

        # context
        context_mask = Tools.get_mask_via_len(context_len, op.context_max_len)  # [B,1,L]
        if context_hidden is None:
            context_hidden = self.main_encoder(context, context_mask)
            context_hidden = context_hidden[0]

        # union
        src_hidden = torch.cat([m_hidden,l_hidden,tp_hidden,context_hidden,related_topic_hidden],1)
        src_mask = torch.cat([m_mask,l_mask,tp_mask,context_mask,related_topics_mask],2)

        # init
        probs = None  
        action_mask = Tools.get_mask_via_len(ar_gth_len, op.action_num)

        if mode == 'train':
            # train
            for i in range(0,op.action_num,2):
                seq_gth = ar_gth[:,0: i+1]
                ar_mask = action_mask[:,:,0:i+1]
                dec_output = Tools._single_decode(seq_gth.detach(), src_hidden, src_mask, self.a_decoder, ar_mask)
                prob = self.proj(dec_output, src_hidden, src_mask, m, l, context,tp_path,related_topics)
                if i == 0:
                    probs = prob
                else:
                    probs = torch.cat([probs, prob], 1)
            return probs
        else:
            # test
            seq_gen = None
            for i in range(0,op.action_num,2):
                if i == 0:
                    seq_gen = ar_gth[:,0:i+1]
                else:
                    seq_gen = torch.cat([seq_gen,ar_gth[:,i:i+1]],1 )
                ar_mask = action_mask[:, :, 0:i + 1]
                dec_output = Tools._single_decode(seq_gen.detach(), src_hidden, src_mask, self.a_decoder,ar_mask)
                single_step_prob = self.proj(dec_output,src_hidden,src_mask,m,l,context,tp_path,related_topics) 
                if i == 0:
                    probs = single_step_prob
                else:
                    probs = torch.cat([probs, single_step_prob], 1)
                single_step_word = torch.argmax(single_step_prob, -1)  # [B,1]
                seq_gen = torch.cat([seq_gen,single_step_word],1)

            return seq_gen, probs 

    def proj(self,dec_out,src_hidden,src_mask,pv_m,l,context,tp,related_topics):
       
        B, L_a = dec_out.size(0), dec_out.size(1)

        
        gen_logit = self.gen_proj(dec_out)

       
        copy_logit = torch.bmm(dec_out, src_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((src_mask == 0).expand(-1, L_a, -1), -1e9)
        logits = torch.cat([gen_logit, copy_logit], -1)

        if op.scale_prj:
            logits *= self.hidden_size ** -0.5

        
        probs = torch.softmax(logits, -1)
        
        gen_prob = probs[:, :, :self.n_topic_vocab]

        # copy from preference
        copy_m_prob =probs[:,:,self.n_topic_vocab :
                               self.n_topic_vocab + op.preference_num]
        copy_m_prob = torch.bmm(copy_m_prob,pv_m) 
        # copy from profile
        copy_l_prob = probs[:, :, self.n_topic_vocab + op.preference_num:
                                  self.n_topic_vocab + op.preference_num + op.profile_num]
        copy_l_prob = torch.bmm(copy_l_prob, l)  

        # copy from context
        copy_context_prob = probs[:, :, self.n_topic_vocab + op.preference_num + op.profile_num + op.state_num:
                                       ]
        transfer_context_word = torch.gather(self.glo2loc.unsqueeze(0).expand(B, -1), 1, context)  # glo_idx to loc_idx
        copy_context_temp = copy_context_prob.new_zeros(B, L_a, self.n_topic_vocab)
        copy_context_prob = copy_context_temp.scatter_add(dim=2,
                                                          index=transfer_context_word.unsqueeze(1).expand(-1, L_a, -1),
                                                          src=copy_context_prob)
        # copy from topic path
        copy_tp_prob = probs[:, :, self.n_topic_vocab + op.preference_num + op.profile_num :
                                    self.n_topic_vocab + op.preference_num + op.profile_num + op.state_num]
        copy_tp_prob = torch.bmm(copy_tp_prob,tp)  

        # copy from related topics
        copy_relation_prob = probs[:,:,self.n_topic_vocab + op.preference_num + op.profile_num + op.context_max_len + op.state_num:]
        copy_relation_prob = torch.bmm(copy_relation_prob,related_topics)

        probs = gen_prob + copy_m_prob + copy_l_prob  + copy_tp_prob + copy_context_prob + copy_relation_prob

        return probs


    def get_movie_rep(self,related_movies,related_movies_mask,p_encoder,relations,relations_len,movie_length):
        
        movie_rep = None
        for i in range(movie_length):
            related_movie = related_movies[:,i]  
            movie_mask = related_movies_mask[:,:,i]
            movie_topic_mask = movie_mask.unsqueeze(-1).expand(-1,-1,op.relation_num)
            related_topics = relations[related_movie,:]   
            topics_len = relations_len[related_movie]
            topics_mask = Tools.get_mask_via_len(topics_len,op.relation_num)  
            topic_mask = movie_topic_mask & topics_mask 
            

            topic_hidden = p_encoder(related_topics,topic_mask)    
            movie_hidden = torch.mean(topic_hidden,1).unsqueeze(1)  

            if movie_rep is None:
                movie_rep = movie_hidden
            else:
                movie_rep = torch.cat([movie_rep,movie_hidden],1)

        return movie_rep

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder

def get_movie_rep(related_movies,related_movies_mask,p_encoder,relations,relations_len,movie_num):
    
    movie_rep = None
    mask = None
    for i in range(movie_num):
        related_movie = related_movies[:,i]  
        movie_mask = related_movies_mask[:,:,i]
        movie_topic_mask = movie_mask.unsqueeze(-1).expand(-1,-1,op.tag_num) 
        related_topics = relations[related_movie,:op.tag_num] 
        topics_len = relations_len[related_movie]
        topics_mask = Tools.get_mask_via_len(topics_len,op.tag_num)  
        topic_mask = movie_topic_mask & topics_mask  

        topic_hidden = p_encoder(related_topics,topic_mask)   
       

        if movie_rep is None:
            movie_rep = topic_hidden
            mask = topic_mask
        else:
            movie_rep = torch.cat([movie_rep,topic_hidden],1)
            mask = torch.cat([mask,topic_mask],2)

    return movie_rep, mask