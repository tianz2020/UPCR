from copy_scheduler import CopyScheduler
import torch
import torch.nn as nn
from tools import  Tools
from option import option as op
import heapq
import functools
import ipdb

class Response(nn.Module):
    def __init__(self,a_encoder,p_encoder,decoder,hidden_size,n_vocab,trg_bos_idx, trg_eos_idx,
                 max_seq_len,main_encoder,beam_width,loc2glo,n_topic):
        super(Response, self).__init__()
        self.main_encoder = main_encoder
        self.a_encoder = a_encoder
        self.p_encoder = p_encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab
        self.n_topic = n_topic
        self.bos_idx = trg_bos_idx
        self.eos_idx = trg_eos_idx
        self.max_len = max_seq_len

        self.beam_width = beam_width
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size, self.n_vocab))
        self.loc2glo = loc2glo

    def forward(self,ar,ar_len,context,context_len,tp_path,tp_path_len,tp_hidden,action_hidden,
                resp_gth=None,resp_gth_len=None):
       
        bs = ar.size(0)
      
        context_mask = Tools.get_mask_via_len(context_len,op.context_max_len)
        context_hidden = self.main_encoder(context,context_mask)

        tp_mask = Tools.get_mask_via_len(tp_path_len, op.state_num)
        tp_hidden = self.p_encoder(tp_path, tp_mask)

        action_mask = Tools.get_mask_via_len(ar_len,op.action_num)
        action_hidden = self.a_encoder(ar,action_mask)

        src_hidden = torch.cat([context_hidden,tp_hidden,action_hidden],1)
        src_mask = torch.cat([context_mask,tp_mask,action_mask],2)

        if resp_gth is not None:
           
            resp_mask = Tools.get_mask_via_len(resp_gth_len,op.r_max_len) & Tools.get_subsequent_mask(resp_gth)
            dec_out = self.decoder(resp_gth,resp_mask,src_hidden,src_mask)
            probs = self.proj(dec_out=dec_out,src_hidden=src_hidden,src_mask=src_mask,context=context,
                                      action=ar,tp=tp_path)
            return probs
        else:
            
            seq_gen = torch.ones(bs, 1, dtype=torch.long) * self.bos_idx
            seq_gen = seq_gen.cuda()

            if op.beam_width == 1:
                seq_gen,probs = self._greedy_search(seq_gen=seq_gen, src_hidden=src_hidden, src_mask=src_mask,
                                             action=ar,context=context,tp=tp_path)
            else:
                                       ar,ar_hidden,ar_mask)
                probs = None
            return seq_gen,probs

    def proj(self,dec_out,src_hidden,src_mask,context,action,tp):
        
        B = action.size(0)

        
        gen_logit = self.gen_proj(dec_out)
        L_r = dec_out.size(1)
        copy_logit = torch.bmm(dec_out, src_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((src_mask == 0).expand(-1, L_r, -1), -1e9)
        logits = torch.cat([gen_logit, copy_logit], -1)  

        if op.scale_prj:
            logits *= self.hidden_size ** -0.5

      
        probs = torch.softmax(logits, -1)
        
        gen_prob = probs[:, :, :self.n_vocab]

        copy_context_prob = probs[:,:,self.n_vocab:
                                      self.n_vocab + op.context_max_len]
        context = one_hot_scatter(context,self.n_vocab)
        copy_context_prob = torch.bmm(copy_context_prob,context)

        copy_tp_prob = probs[:,:,self.n_vocab + op.context_max_len:
                                 self.n_vocab + op.context_max_len + op.state_num]
        transfer_tp_word = torch.gather(self.loc2glo.unsqueeze(0).expand(B, -1), 1, tp)
        copy_tp_temp = copy_tp_prob.new_zeros(B, L_r, self.n_vocab)
        copy_tp_prob = copy_tp_temp.scatter_add(dim=2,
                                                index=transfer_tp_word.unsqueeze(1).expand(-1, L_r, -1),
                                                src=copy_tp_prob)

        copy_ar_prob = probs[:, :, self.n_vocab + op.context_max_len + op.state_num:
                                   ]
        transfer_ar_word = torch.gather(self.loc2glo.unsqueeze(0).expand(B, -1), 1, action)
        copy_ar_temp = copy_ar_prob.new_zeros(B, L_r, self.n_vocab)
        copy_ar_prob = copy_ar_temp.scatter_add(dim=2,
                                                index=transfer_ar_word.unsqueeze(1).expand(-1, L_r, -1),
                                                src=copy_ar_prob)


        probs = gen_prob + copy_context_prob + copy_tp_prob + copy_ar_prob
        return probs

    def _greedy_search(self, seq_gen, src_hidden, src_mask,
                       action,context,tp):
        probs = None
        for step in range(op.r_max_len):
            
            single_step_probs = self.single_decode(input_seq=seq_gen,src_hidden=src_hidden,src_mask=src_mask,
                                                   decoder=self.decoder,action=action,context=context,tp=tp) 
            if probs is None:
                probs = single_step_probs
            else:
                probs = torch.cat([probs,single_step_probs],1)
            single_step_word = torch.argmax(single_step_probs, -1)
            
            seq_gen = torch.cat([seq_gen, single_step_word], 1) 

        return seq_gen[:,1:],probs

    def single_decode(self, input_seq, src_hidden, src_mask, decoder,
                       action,context,tp):
      
        dec_output = Tools._single_decode(input_seq.detach(), src_hidden, src_mask, decoder)
        
        single_step_probs = self.proj(dec_out=dec_output,src_hidden=src_hidden,src_mask=src_mask,context=context,
                                      action=action,tp=tp)
        return single_step_probs


    def _beam_search(self, seq_gen, src_hiddens, src_mask,
                     context_hidden, context_mask, context,
                     ar, ar_hidden,ar_mask , tp, tp_hidden, tp_mask):

        batch_size = seq_gen.size(0)
        scores = torch.zeros(batch_size * self.beam_width, dtype=torch.float).cuda() 
        mature_buckets = [MatureBucket(self.beam_width) for _ in range(batch_size)]

        
        for i in range(op.r_max_len-1):
            if i == 0:
              
                i_step_output = self.single_decode(seq_gen, src_hiddens, src_mask, self.decoder,
                                                   ar,pv_ru)
                
                topk_probs, word_index = i_step_output.topk(self.beam_width, dim=-1)

                scores = scores + torch.log(topk_probs.reshape(-1)) 
                flat_word_index = word_index.reshape(-1, 1)

                seq_gen = expand_if_not_none(seq_gen, 0, self.beam_width)  
                seq_gen = torch.cat([seq_gen, flat_word_index], 1) 

            
                src_hiddens = expand_if_not_none(src_hiddens, 0, self.beam_width)  
                src_mask = expand_if_not_none(src_mask, 0, self.beam_width)
                pv_ru_hidden = expand_if_not_none(pv_ru_hidden, 0, self.beam_width)
                pv_ru_mask = expand_if_not_none(pv_ru_mask, 0, self.beam_width)
                pv_ru = expand_if_not_none(pv_ru, 0, self.beam_width)
                ar_hidden = expand_if_not_none(ar_hidden, 0, self.beam_width)
                ar = expand_if_not_none(ar, 0, self.beam_width)
            else:
                
                i_step_output = self.single_decode(seq_gen, src_hiddens, src_mask, self.decoder,
                                                   ar,pv_ru)
               
                topk_probs, word_index = i_step_output.topk(self.beam_width, dim=-1)
                topk_probs = topk_probs.reshape(batch_size, -1)  
                scores = scores.unsqueeze(-1).expand(-1, self.beam_width).reshape(batch_size, -1) + topk_probs
                rets = self.harvest(scores, seq_gen, word_index, batch_size)

                if rets is not None:
                    scores, harvest_info = rets
                    for bi, gain in harvest_info:
                        mature_buckets[bi].push(gain)

                
                topk_probs, topk_indices = scores.topk(self.beam_width, 1)
                scores = topk_probs.reshape(-1)  # B * k

               
                expand_seq_gen = seq_gen.unsqueeze(1).expand(-1, self.beam_width, -1)
                
                permute_word_output = word_index.permute(0, 2, 1)
                
                seq_gen = torch.cat([expand_seq_gen, permute_word_output], dim=2)
                
                seq_gen = seq_gen.reshape(batch_size, self.beam_width ** 2, -1)
                
                seq_gen = nested_index_select(seq_gen, topk_indices.long()).reshape(batch_size * self.beam_width, -1)

        scores, bst_trajectory_index = scores.reshape(batch_size, self.beam_width).max(-1)
        scores = scores.detach().cpu().numpy().tolist()
        
        bst_trajectory = nested_index_select(seq_gen.reshape(batch_size, self.beam_width, -1),
                                             bst_trajectory_index.unsqueeze(-1).long()).squeeze(1)
        for i, s in enumerate(scores):
            traj = bst_trajectory[i]
            mature_buckets[i].push(Branch(s, traj, op.r_beam_max_len))

       
        bst_trajectory = torch.stack([mb.get_max() for mb in mature_buckets], dim=0)
        return bst_trajectory

    def harvest(self, scores, history, word_index, obs):
        
        word_index = word_index.reshape(obs, -1)
        eos_sign = (word_index == self.eos_idx)
        eos_num = eos_sign.long().sum().item()
        if eos_num <= 0:
            return None

        _, eos_indices = eos_sign.long().reshape(-1).sort(descending=True)
        eos_scores = scores.reshape(-1).index_select(0, eos_indices).cpu().numpy().tolist()

        eos_indices = eos_indices[:eos_num]
        eos_x = (eos_indices / op.beam_width).long()
        batch_index = ((eos_indices / (op.beam_width * op.beam_width)).long()).cpu().numpy().tolist()
        mature_traj = history[eos_x].clone()
        scores = scores.masked_fill(eos_sign, -1e20)
        grow_len = mature_traj.size(1)

        if grow_len < op.r_beam_max_len + 1:
            padding = mature_traj.new_ones(eos_num, op.r_beam_max_len - grow_len + 1,
                                           dtype=torch.long) * self.eos_idx
            mature_traj = torch.cat([mature_traj, padding], dim=-1)

        return scores, [(batch_index[i], Branch(eos_scores[i], mature_traj[i, :], grow_len)) for i in range(eos_num)]

def nested_index_select(origin_data, select_index):
    origin_data_shape = list(origin_data.shape)
    select_index_shape = list(select_index.shape)

    work_axes = len(select_index_shape) - 1
    grad_v = functools.reduce(lambda x, y: x * y, origin_data_shape[:work_axes])
    new_dim = select_index_shape[-1]
    grad = torch.arange(0, grad_v, dtype=torch.long, device=origin_data.device).unsqueeze(-1)
    grad = grad.expand(-1, new_dim)
    grad = grad.reshape(-1)
    grad = grad * origin_data_shape[work_axes]
    select_index = select_index.reshape(-1) + grad
    reshaped_data = origin_data.reshape(grad_v * origin_data_shape[work_axes], -1)
    selected_data = reshaped_data.index_select(0, select_index)
    origin_data_shape[work_axes] = new_dim
    selected_data = selected_data.reshape(origin_data_shape)
    return selected_data

def expand_if_not_none(tensor, dim, beam_width):
    
    if tensor is None:
        return None
    tensor_shape = list(tensor.shape)
    tensor = tensor.unsqueeze(dim + 1)
    expand_dims = [-1] * (len(tensor_shape) + 1)
    expand_dims[dim + 1] = beam_width
    tensor = tensor.expand(*expand_dims)
    tensor_shape[dim] = tensor_shape[dim] * beam_width
    tensor = tensor.reshape(*tensor_shape)
    return tensor.contiguous()

class Branch:
    def __init__(self, score, tensor, length, alpha=1.0, log_act=True):
        self.score = Branch.normal_score(score, length, alpha, log_act)
        self.tensor = tensor

    def __lt__(self, other):
        return self.score <= other.score

    def __eq__(self, other):
        return self.score == other.score

    def __gt__(self, other):
        return self.score >= other.score

    @staticmethod
    def normal_score(score, length, alpha=1.0, log_act=True):
        assert alpha >= 0.0, "alpha should >= 0.0"
        assert alpha <= 1.0, "alpha should <= 1.0"

        if log_act:
            score = score / (length ** alpha)
        else:
            score = score ** (1 / (length ** alpha))

        return score

    def get_tensor(self):
        return self.tensor

class MatureBucket:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.bucket = []

    def push(self, item: Branch):
        if len(self.bucket) < self.bucket_size:
            heapq.heappush(self.bucket, item)
        else:
            if item.score > self.bucket[0].score:
                heapq.heappushpop(self.bucket, item)

    def get_max(self):
        self.bucket = sorted(self.bucket, reverse=True)
        return self.bucket[0].get_tensor()


def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder