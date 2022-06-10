from gumbel_softmax import GumbelSoftmax
from tau_scheduler import TauScheduler
import torch
import torch.nn as nn
from tools import Tools
from option import option as op

class PriorPreference(nn.Module):
    def __init__(self,encoder, decoder,hidden_size, n_topic_vocab,
                 trg_bos_idx, max_seq_len,glo2loc,loc2glo,main_tfr_encoder,
                 gs: GumbelSoftmax, ts: TauScheduler):
        super(PriorPreference, self).__init__()
        self.decoder = decoder
        self.p_encoder = encoder
        self.main_tfr_encoder = main_tfr_encoder
        self.n_topic_vocab = n_topic_vocab
        self.bos_idx = trg_bos_idx
        self.max_seq_len = max_seq_len
        self.gs = gs
        self.ts = ts
        self.glo2loc = glo2loc
        self.loc2glo = loc2glo
        self.hidden_size = hidden_size
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size, self.n_topic_vocab))

    def forward(self,context,context_len,pv_m,pv_m_mask,tp_path,tp_path_len):
        bs = pv_m.size(0)
        pv_m_hidden = self.p_encoder(pv_m,pv_m_mask)
        context_mask = Tools.get_mask_via_len(context_len, op.context_max_len)
        context_hidden = self.main_tfr_encoder(context, context_mask)
        tp_path = one_hot_scatter(tp_path,self.n_topic_vocab)
        tp_mask = Tools.get_mask_via_len(tp_path_len,op.state_num)
        tp_hidden = self.p_encoder(tp_path,tp_mask)
        src_hiddens = torch.cat([pv_m_hidden,tp_hidden,context_hidden], 1)
        src_mask = torch.cat([pv_m_mask,tp_mask,context_mask], 2)
        seq_gen_gumbel = Tools._generate_init(bs, self.n_topic_vocab, trg_bos_idx=self.bos_idx,training=self.training)
        seq_gen_prob = None
        for _ in range(op.preference_num):
            dec_output = Tools._single_decode(seq_gen_gumbel.detach(),src_hiddens,src_mask,self.decoder)
            single_step_prob = self.proj(dec_out=dec_output,context=context,src_hidden=src_hiddens,
                                         src_mask=src_mask,pv_m=pv_m,tp_path=tp_path)
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

    def proj(self,dec_out,context,src_hidden,src_mask,pv_m,tp_path):
        gen_logit = self.gen_proj(dec_out)
        L_s = dec_out.size(1)
        B = context.size(0)
        copy_logit = torch.bmm(dec_out, src_hidden.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((src_mask == 0).expand(-1, L_s, -1), -1e9)
        logits = torch.cat([gen_logit, copy_logit], -1)
        if op.scale_prj:
            logits *= self.hidden_size ** -0.5
        probs = torch.softmax(logits, -1)
        gen_prob = probs[:, :, :self.n_topic_vocab]
        copy_pv_m_prob = probs[:, :, self.n_topic_vocab:self.n_topic_vocab + op.preference_num ]
        copy_pv_m_prob = torch.bmm(copy_pv_m_prob, pv_m)
        copy_context_prob = probs[:,:,self.n_topic_vocab + op.preference_num + op.state_num:]
        transfer_context_word = torch.gather(self.glo2loc.unsqueeze(0).expand(B, -1),1, context)
        copy_context_temp = copy_context_prob.new_zeros(B, L_s, self.n_topic_vocab)
        copy_context_prob = copy_context_temp.scatter_add(dim=2,index=transfer_context_word.unsqueeze(1).expand(-1, L_s, -1),
                                                          src=copy_context_prob)
        copy_tp_prob = probs[:, :, self.n_topic_vocab + op.preference_num:self.n_topic_vocab + op.preference_num + op.state_num]
        copy_tp_prob = torch.bmm(copy_tp_prob, tp_path)
        probs = gen_prob + copy_pv_m_prob + copy_tp_prob + copy_context_prob
        return probs

class PosteriorPreference(nn.Module):
    def __init__(self,encoder,decoder,main_encoder, hidden_size, n_topic_vocab,glo2loc,loc2glo,
                 trg_bos_idx, max_seq_len, gs: GumbelSoftmax, ts: TauScheduler):
        super(PosteriorPreference, self).__init__()
        self.p_encoder = encoder
        self.decoder = decoder
        self.main_encoder = main_encoder
        self.glo2loc = glo2loc
        self.loc2glo = loc2glo
        self.hidden_size = hidden_size
        self.n_topic_vocab = n_topic_vocab
        self.trg_bos_idx = trg_bos_idx
        self.max_seq_len = max_seq_len
        self.gs = gs
        self.ts = ts
        self.gen_proj = nn.Sequential(nn.Linear(self.hidden_size, self.n_topic_vocab))

    def forward(self,context,context_len,pv_m,pv_m_mask,ar_gth,ar_gth_len,tp_path,tp_path_len):
        bs = pv_m.size(0)
        ar_gth_len = [ int(length/2) for length in ar_gth_len]
        ar_gth_len = torch.tensor(ar_gth_len).cuda()
        ar_gth = ar_gth[:,[1,3,5,7,9]]
        ar_gth = one_hot_scatter(ar_gth,self.n_topic_vocab)
        ar_mask = Tools.get_mask_via_len(ar_gth_len, int(op.action_num/2))
        ar_hidden = self.p_encoder(ar_gth, ar_mask)
        context_mask = Tools.get_mask_via_len(context_len, op.context_max_len)
        context_hidden = self.main_encoder(context, context_mask)
        pv_m_hidden = self.p_encoder(pv_m, pv_m_mask)
        tp_path = one_hot_scatter(tp_path, self.n_topic_vocab)
        tp_mask = Tools.get_mask_via_len(tp_path_len, op.state_num)
        tp_hidden = self.p_encoder(tp_path, tp_mask)
        src_hiddens = torch.cat([pv_m_hidden,tp_hidden,context_hidden,ar_hidden], 1)
        src_mask = torch.cat([pv_m_mask,tp_mask,context_mask,ar_mask], 2)
        seq_gen_gumbel = Tools._generate_init(bs, self.n_topic_vocab, trg_bos_idx=self.trg_bos_idx)
        seq_gen_prob = None
        for _ in range(op.preference_num):
            dec_output = Tools._single_decode(seq_gen_gumbel.detach(), src_hiddens, src_mask, self.decoder)
            single_step_prob = self.proj(dec_out=dec_output,src_hidden=src_hiddens,src_mask=src_mask,
                                         pv_m=pv_m,context=context,tp=tp_path)
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

    def proj(self, dec_out,  src_hidden, src_mask, pv_m, context,tp):
        B, L_s = dec_out.size(0), dec_out.size(1)
        gen_logit = self.gen_proj(dec_out)
        hidden_no_At = src_hidden[:,0:-5,:]
        mask_no_At = src_mask[:,:,0:-5]
        copy_logit = torch.bmm(dec_out, hidden_no_At.permute(0, 2, 1))
        copy_logit = copy_logit.masked_fill((mask_no_At == 0).expand(-1, L_s, -1), -1e9)
        logits = torch.cat([gen_logit, copy_logit], -1)
        if op.scale_prj:
            logits *= self.hidden_size ** -0.5
        probs = torch.softmax(logits, -1)
        gen_prob = probs[:, :, :self.n_topic_vocab]
        copy_pv_m_prob = probs[:, :, self.n_topic_vocab:self.n_topic_vocab + op.preference_num]
        copy_pv_m_prob = torch.bmm(copy_pv_m_prob, pv_m)
        copy_tp_path_prob = probs[:, :,self.n_topic_vocab + op.preference_num :self.n_topic_vocab + op.preference_num + op.state_num]
        copy_tp_path_prob = torch.bmm(copy_tp_path_prob, tp)
        copy_context_prob = probs[:, :, self.n_topic_vocab + op.preference_num + op.state_num:]
        transfer_context_word = torch.gather(self.glo2loc.unsqueeze(0).expand(B, -1), 1, context)
        copy_context_temp = copy_context_prob.new_zeros(B, L_s, self.n_topic_vocab)
        copy_context_prob = copy_context_temp.scatter_add(dim=2,index=transfer_context_word.unsqueeze(1).expand(-1, L_s, -1),src=copy_context_prob)
        probs = gen_prob + copy_pv_m_prob + copy_tp_path_prob + copy_context_prob
        return probs

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder