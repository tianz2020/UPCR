import torch
import torch.nn as nn
from tools import  Tools
from option import option as op

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

    def forward(self,ar,ar_len,context,context_len,tp_path,tp_path_len,
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
            probs = self.proj(dec_out=dec_out,src_hidden=src_hidden,src_mask=src_mask,context=context,action=ar,tp=tp_path)
            return probs
        else:
            seq_gen = torch.ones(bs, 1, dtype=torch.long) * self.bos_idx
            seq_gen = seq_gen.cuda()
            seq_gen,probs = self._greedy_search(seq_gen=seq_gen, src_hidden=src_hidden, src_mask=src_mask,action=ar,context=context,tp=tp_path)
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
        copy_context_prob = probs[:,:,self.n_vocab:self.n_vocab + op.context_max_len]
        context = one_hot_scatter(context,self.n_vocab)
        copy_context_prob = torch.bmm(copy_context_prob,context)
        copy_tp_prob = probs[:,:,self.n_vocab + op.context_max_len:self.n_vocab + op.context_max_len + op.state_num]
        transfer_tp_word = torch.gather(self.loc2glo.unsqueeze(0).expand(B, -1), 1, tp)
        copy_tp_temp = copy_tp_prob.new_zeros(B, L_r, self.n_vocab)
        copy_tp_prob = copy_tp_temp.scatter_add(dim=2,index=transfer_tp_word.unsqueeze(1).expand(-1, L_r, -1),src=copy_tp_prob)
        copy_ar_prob = probs[:, :, self.n_vocab + op.context_max_len + op.state_num:]
        transfer_ar_word = torch.gather(self.loc2glo.unsqueeze(0).expand(B, -1), 1, action)
        copy_ar_temp = copy_ar_prob.new_zeros(B, L_r, self.n_vocab)
        copy_ar_prob = copy_ar_temp.scatter_add(dim=2,index=transfer_ar_word.unsqueeze(1).expand(-1, L_r, -1),src=copy_ar_prob)
        probs = gen_prob + copy_context_prob + copy_tp_prob + copy_ar_prob
        return probs

    def _greedy_search(self, seq_gen, src_hidden, src_mask,
                       action,context,tp):
        probs = None
        for step in range(op.r_max_len):
            single_step_probs = self.single_decode(input_seq=seq_gen,src_hidden=src_hidden,src_mask=src_mask,decoder=self.decoder,action=action,context=context,tp=tp)
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

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder