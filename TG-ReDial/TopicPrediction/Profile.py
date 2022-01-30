import logging

from gumbel_softmax import GumbelSoftmax
from tau_scheduler import TauScheduler
import torch.nn as nn
import torch
from option import option as op
from tools import Tools
import ipdb

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

    def proj(self,dec_out,topics_hidden,topics,topics_mask):
        
   
        gen_logit = self.gen_proj(dec_out)
        L_s = dec_out.size(1)

        
        copy_logit = torch.bmm(dec_out, topics_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((topics_mask == 0).expand(-1, L_s, -1), -1e9)
        
        logits = torch.cat([gen_logit, copy_logit], -1)

        if op.scale_prj:
            logits *= self.hidden_size ** -0.5

        
        probs = torch.softmax(logits, -1)
        
        gen_prob = probs[:, :, :self.n_topic_vocab]

        copy_topic_prob = probs[:,:,self.n_topic_vocab:]
        copy_topic_prob = torch.bmm(copy_topic_prob,topics)

        probs = gen_prob + copy_topic_prob
        return probs

    def get_movie_rep(self,related_movies,related_movies_mask,p_encoder,relations,relations_len):
        
        movie_rep = None
        mask = None
        for i in range(op.movie_path_len):
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


def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder
