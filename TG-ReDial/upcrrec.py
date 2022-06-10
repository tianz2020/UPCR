import torch.nn as nn
import torch
from option import  option as op
from tau_scheduler import  TauScheduler
from copy_scheduler import CopyScheduler
from transformer.Models import Encoder
from transformer.Models import Decoder
from Profile import PriorProfile,PosteriorProfile
from Preference_all import PriorPreference,PosteriorPreference
from gumbel_softmax import GumbelSoftmax
from Action_all import Action
from tools import Tools
from Vocab import Vocab
from scipy import optimize
import torch.nn.functional as F
from transformer.Optim import ScheduledOptim
from DataLoaderRec import  DataLoaderRec
import math
import sys

class Upcrrec(nn.Module):
    def __init__(self,vocab:Vocab,user_cont,n_layers=6,p_layers=3,
                 d_word_vec=512,d_model=512, d_inner=2048,beam_width=1,
                 n_head=8, d_k=64, d_v=64, dropout=0.1):

        super(Upcrrec, self).__init__()
        self.vocab = vocab
        self.glo2loc , self.loc2glo = vocab.vocab_transfer()
        self.glo2loc = torch.tensor(self.glo2loc).cuda()
        self.loc2glo = torch.tensor(self.loc2glo).cuda()
        self.topic_num = vocab.topic_num()
        self.word_vocab, self.word_len, self.topic_vocab, self.topic_len = vocab.get_vocab(task='rec')
        self.word_pad_idx = vocab.get_word_pad()
        self.topic_pad_idx = vocab.get_topic_pad()
        self.m_bos_idx = vocab.topic2index(op.BOS_PRE)
        self.l_bos_idx = vocab.topic2index(op.BOS_PRO)
        self.a_bos_idx = vocab.topic2index(op.BOS_ACTION)
        self.r_bos_idx = vocab.word2index(op.BOS_RESPONSE)
        self.r_eos_idx = vocab.word2index(op.EOS_RESPONSE)
        self.beam_width = beam_width
        self.pro_tau_scheduler = TauScheduler(op.init_tau, op.tau_mini, op.tau_decay_total_steps)
        self.pre_tau_scheduler = TauScheduler(op.init_tau, op.tau_mini, op.tau_decay_total_steps)
        self.m_copy_scheduler = CopyScheduler(op.s_copy_lambda, op.copy_lambda_mini, op.copy_lambda_decay_steps)
        self.l_copy_scheduler = CopyScheduler(op.a_copy_lambda, op.copy_lambda_mini, op.copy_lambda_decay_steps)
        self.word_emb = nn.Embedding(self.word_len,d_word_vec,padding_idx=self.word_pad_idx)
        self.topic_emb = nn.Embedding(self.topic_len,d_word_vec,padding_idx=self.topic_pad_idx)
        self.user_emb = nn.Embedding(user_cont,d_word_vec)
        self.gumbel_softmax = GumbelSoftmax()
        self.global_step = 0
        self.main_tfr_encoder = Encoder(
            n_src_vocab=self.word_len, n_position=op.conv_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.word_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.word_emb
        )

        self.u_tfr_encoder4p = Encoder(
            n_src_vocab=user_cont, n_position=1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.user_emb
        )

        self.u_tfr_encoder4q = Encoder(
            n_src_vocab=user_cont, n_position=1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.user_emb
        )

        self.p_tfr_encoder4p = Encoder(
            n_src_vocab=self.topic_len, n_position=200,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )
        self.p_tfr_decoder4p = Decoder(
            n_trg_vocab=self.topic_len, n_position=15,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )

        self.m_tfr_encoder4p = Encoder(
            n_src_vocab=self.topic_len, n_position=100,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )

        self.p_tfr_encoder4q = Encoder(
            n_src_vocab=self.topic_len, n_position=200,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )
        self.p_tfr_decoder4q = Decoder(
            n_trg_vocab=self.topic_len, n_position=15,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )
        self.m_tfr_encoder4q = Encoder(
            n_src_vocab=self.topic_len, n_position=100,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )

        self.a_tfr_decoder = Decoder(
            n_trg_vocab=self.topic_len, n_position=op.action_num,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )

        if op.wo_l:
            self.p_l = None
            self.q_l = None
        else:
            self.p_l = PriorProfile(encoder=self.u_tfr_encoder4p,decoder=self.p_tfr_decoder4p,
                                    hidden_size=d_model,n_topic_vocab=self.topic_len,
                                    trg_bos_idx=self.l_bos_idx,max_seq_len=op.profile_num,
                                    gs=self.gumbel_softmax,ts=self.pro_tau_scheduler).cuda()

            self.q_l = PosteriorProfile(main_encoder=self.main_tfr_encoder,id_encoder = self.u_tfr_encoder4q,
                                    topic_encoder=self.p_tfr_encoder4q, decoder=self.p_tfr_decoder4q,m_encoder=self.m_tfr_encoder4q,
                                    hidden_size=d_model,n_topic_vocab=self.topic_len,
                                    trg_bos_idx=self.l_bos_idx,max_seq_len=op.profile_num,
                                    gs=self.gumbel_softmax,ts=self.pro_tau_scheduler,
                                    glo2loc=self.glo2loc,loc2glo=self.loc2glo).cuda()

        if op.wo_m:
            self.p_mt = None
            self.q_mt = None

        else:
            self.p_mt = PriorPreference(encoder=self.p_tfr_encoder4p,decoder=self.p_tfr_decoder4p,
                                       m_encoder=self.m_tfr_encoder4p,main_tfr_encoder=self.main_tfr_encoder,
                                       hidden_size=d_model,n_topic_vocab=self.topic_len,trg_bos_idx=self.m_bos_idx,
                                       max_seq_len=op.preference_num,gs=self.gumbel_softmax,glo2loc=self.glo2loc,
                                       loc2glo=self.loc2glo,ts=self.pre_tau_scheduler).cuda()

            self.q_mt = PosteriorPreference(encoder=self.p_tfr_encoder4q,main_encoder=self.main_tfr_encoder,
                                           m_encoder=self.m_tfr_encoder4q,decoder=self.p_tfr_decoder4q,
                                         hidden_size=d_model,n_topic_vocab=self.topic_len,trg_bos_idx=self.m_bos_idx,
                                         max_seq_len=op.preference_num,gs=self.gumbel_softmax,glo2loc=self.glo2loc,
                                         loc2glo=self.loc2glo,ts=self.pre_tau_scheduler).cuda()

        self.action = Action(p_encoder=self.p_tfr_encoder4p,main_encoder=self.main_tfr_encoder,m_encoder=self.m_tfr_encoder4p,
                             a_decoder=self.a_tfr_decoder,hidden_size=d_model,
                             n_topic_vocab=self.topic_len,bos_idx=self.a_bos_idx,vocab=self.vocab,
                             max_len=op.action_num,glo2loc=self.glo2loc,loc2glo=self.loc2glo).cuda()

    def forward(self,user_id,all_topic, all_topic_len,context, context_len,tp_path, tp_path_len,
                ar_gth, ar_gth_len,related_movies, related_movies_len,final,pv_m,mode='train'):
        assert mode in ['train','valid','test']
        pv_m, pv_m_mask = self.mask_preference(pv_m, final)
        if mode == 'train':
            p_l, p_l_gumbel = self.p_l.forward(id=user_id)
            q_l, q_l_gumbel = self.q_l.forward(id=user_id, topics=all_topic, topics_len=all_topic_len)
            p_m, p_m_gumbel = self.p_mt.forward(context=context, context_len=context_len,pv_m=pv_m,pv_m_mask=pv_m_mask,
                                                tp_path=tp_path,tp_path_len=tp_path_len)
            q_m, q_m_gumbel = self.q_mt.forward(context=context, context_len=context_len,pv_m=pv_m,pv_m_mask=pv_m_mask,
                                                ar_gth=ar_gth,ar_gth_len=ar_gth_len,tp_path=tp_path,tp_path_len=tp_path_len)
            ar = self.action.forward(m=q_m_gumbel,
                                     l=q_l_gumbel,
                                     context=context,
                                     context_len=context_len,
                                     ar_gth=ar_gth,ar_gth_len=ar_gth_len,
                                     tp_path=tp_path,tp_path_len=tp_path_len,
                                     related_movies=related_movies,related_movies_len=related_movies_len,
                                     mode='train')
            self.global_step += 1
            return p_l, q_l, p_m, q_m, ar, q_m_gumbel
        else:
            p_l = self.p_l.forward(id=user_id)
            p_m = self.p_mt.forward(context=context,context_len=context_len,pv_m=pv_m,pv_m_mask=pv_m_mask,tp_path=tp_path,tp_path_len=tp_path_len)
            ar,ar_probs = self.action.forward(m=p_m,l=p_l,context=context,context_len=context_len,ar_gth=ar_gth,ar_gth_len=ar_gth_len,tp_path=tp_path,
                                              tp_path_len=tp_path_len,related_movies=related_movies, related_movies_len=related_movies_len,mode='test')
            return  ar, ar_probs, p_m

    def mask_preference(self, pv_m, final):
        b = range(op.batch_size)
        b = [i + 1 for i in b]
        b = torch.tensor(b).cuda().tolist()
        final = list(final)
        final = [int(i) for i in final]
        c = [i * j for i, j in zip(final, b)]
        c = list(c)
        d = []
        for i in c:
            if i != 0:
                d.append(c.index(i))
        if d:
            d = torch.tensor(d).cuda()
            pv_m[d, :, :] = 0
            pv_m[d, :, self.topic_pad_idx] = 1.0
        pv_m_mask = pv_m.new_ones(pv_m.size(0), 1, pv_m.size(1))
        pv_m_mask[d,:,:] = 0
        return pv_m, pv_m_mask

    def topictensor2nl(self,tensor):
        words = tensor.detach().cpu().numpy()
        words = self.vocab.index2topic(words)
        return words

    def wordtensor2nl(self,tensor):
        words = tensor.detach().cpu().numpy()
        words = self.vocab.index2word(words)
        return words

class Engine():
    def __init__(self,model:torch.nn.Module,
                 vocab):
        self.model = model
        lr = op.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = ScheduledOptim(self.optimizer, 0.5, op.d_model, op.n_warmup_steps)
        self.vocab = vocab
        self.topic_pad_idx = self.vocab.topic2index(op.PAD_WORD)
        self.global_step = 0
        self.action_loss = 0
        self.kl_l_loss = 0
        self.kl_m_loss =0

    def train(self,train_set,test_set):
        bst_metric = 0
        patience = 0
        gen_stop = False
        for e in range(op.epoch):
            print("epoch : {}".format(e))
            train_loader = DataLoaderRec(train_set,self.vocab)
            self.pv_m = get_default_tensor([op.batch_size, op.preference_num, self.model.topic_len], torch.float,pad_idx=self.topic_pad_idx)
            self.optimizer.zero_grad()
            for index,input in enumerate(train_loader):
                if input[0].size(0) != op.batch_size:
                    break
                id, all_topic, all_topic_len,context_idx, context_len, topic_path, topic_path_len, a_R, a_R_len, \
                seek_idx, seek_len, resp_idx, resp_len, state_R, state_R_len, related_movies,related_movies_len,final = input
                p_l, q_l, p_m, q_m, ar, m= self.model.forward(user_id=id,all_topic=all_topic,all_topic_len=all_topic_len,context=context_idx, context_len=context_len,
                                                              tp_path=topic_path,tp_path_len=topic_path_len,ar_gth=a_R, ar_gth_len=a_R_len,
                                                              related_movies=related_movies, related_movies_len=related_movies_len,final=final,pv_m=self.pv_m)
                kl_l = kl_loss(p_l, q_l.detach())
                self.kl_l_loss += kl_l.item()
                kl_m = kl_loss(p_m, q_m.detach())
                self.kl_m_loss += kl_m.item()
                nll_ar = action_nll(ar, a_R.detach(), self.model.topic_pad_idx)
                self.action_loss += nll_ar.item()
                p_l_reg, q_l_reg = regularization_loss(p_l), regularization_loss(q_l)
                p_m_reg, q_m_reg = regularization_loss(p_m), regularization_loss(q_m)
                reg_loss = op.reg_lambda * (p_l_reg + q_l_reg + p_m_reg + q_m_reg)
                loss = 0.3 * kl_m + 0.3 * kl_l + nll_ar + reg_loss
                if (self.global_step % 200 == 0):
                    print("global_step: {}".format(self.global_step))
                    print("kl_preference: {}".format(self.kl_m_loss / self.model.global_step))
                    print("kl_profile: {}".format(self.kl_l_loss / self.model.global_step))
                    print("nll_ar: {}".format(self.action_loss / self.model.global_step))
                    sys.stdout.flush()
                loss = loss / float(op.gradient_stack)
                loss.backward(retain_graph=False)
                if self.global_step % op.gradient_stack == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.pv_m = m.detach()
                self.global_step += 1
            metric = self.test(test_set)
            if bst_metric > metric['recall@10']:
                patience += 1
                print(f"[Patience = {patience}]")
            else:
                patience = 0
                bst_metric = metric['recall@10']
        print("train finished ! ")

    def test(self,test_set):
        self.model.eval()
        print(" test ")
        dataloader = DataLoaderRec(test_set,self.vocab)
        metrics = {
            "NDCG1": 0,
            "NDCG10": 0,
            "NDCG50": 0,
            "MRR1": 0,
            "MRR10": 0,
            "MRR50": 0,
            "rec_count": 0,
            "count":0
        }
        self.pv_m = get_default_tensor([op.batch_size, op.preference_num, self.model.topic_len], torch.float,pad_idx=self.model.topic_pad_idx)
        with torch.no_grad():
            for index,data in enumerate(dataloader):
                if data[0].size(0) != op.batch_size:
                    break
                id, all_topic, all_topic_len, context_idx, context_len, topic_path, topic_path_len, a_R, a_R_len, \
                seek_idx, seek_len, resp_idx, resp_len, state_R, state_R_len, related_movies, related_movies_len, final = data
                ar, ar_probs, m = self.model.forward(user_id=id,all_topic=all_topic,all_topic_len=all_topic_len,context=context_idx, context_len=context_len,
                                                     tp_path=topic_path,tp_path_len=topic_path_len,ar_gth=a_R, ar_gth_len=a_R_len,
                                                     related_movies=related_movies, related_movies_len=related_movies_len,final=final,pv_m=self.pv_m,mode='test')
                self.pv_m = one_hot_scatter(m,self.vocab.topic_num())
                self.compute_metrics(ar_probs, a_R, a_R_len, metrics)
        metrics['NDCG1'] = round(metrics['NDCG1'] / metrics['rec_count'], 4)
        metrics['NDCG10'] = round(metrics['NDCG10'] / metrics['rec_count'], 4)
        metrics['NDCG50'] = round(metrics['NDCG50'] / metrics['rec_count'], 4)
        metrics['MRR1'] = round(metrics['MRR1'] / metrics['rec_count'], 4)
        metrics['MRR10'] = round(metrics['MRR10'] / metrics['rec_count'], 4)
        metrics['MRR50'] = round(metrics['MRR50'] / metrics['rec_count'], 4)
        print(metrics)
        self.model.train()
        print('test finished!')
        return metrics

    def compute_metrics(self,ar_probs, ar_gth, a_R_len, metrics):
        tanlun = self.vocab.topic2index('谈论')
        qingqiutuijian = self.vocab.topic2index('请求推荐')
        def _topic_prediction(tar,gen,metrics):
            metrics['topic_count'] += 1
            for k in [1,3,5]:
                pred, pred_id = torch.topk(gen,k,-1)
                pred_id = pred_id.tolist()
                if tar in pred_id:
                    metrics["TopicId_Hits@{}".format(k)] += 1
        def _movie_recommendation(tar,gen,metrics):
            _, pred_idx = torch.topk(gen, k=100, dim=0)
            metrics["count"] += 1
            metrics['rec_count'] += 1
            for k in [1,10,50]:
                pred, pred_id = torch.topk(gen,k,-1)
                pred_id = pred_id.tolist()
                if tar in pred_id:
                    rank = pred_id.index(tar)
                    metrics['NDCG{}'.format(k)] += 1.0 / math.log(rank + 2.0, 2)
                    metrics['MRR{}'.format(k)] += 1.0 / (rank + 1.0)
        for i, gt in enumerate(ar_gth):
            ar_gen = ar_probs[i,:]
            gt_len = int(a_R_len[i])
            for j in range(0,gt_len,2):
                action_type = gt[j]
                if action_type == self.vocab.topic2index('推荐电影'):
                    _movie_recommendation(gt[j+1],ar_gen[int(j/2)],metrics)
                else:
                    _topic_prediction(gt[j+1],ar_gen[int(j/2)],metrics)
                    if tanlun in gt and qingqiutuijian in gt:
                        break

def get_mask_via_len(length, max_len):
    B = length.size(0)
    mask = torch.ones([B, max_len]).cuda()
    mask = torch.cumsum(mask, 1)
    mask = mask <= length.unsqueeze(1)
    mask = mask.unsqueeze(-2)
    return mask

def get_default_tensor(shape, dtype, pad_idx=None):
    pad_tensor = torch.zeros(shape, dtype=dtype)
    pad_tensor[..., pad_idx] = 1.0 if dtype == torch.float else 1
    pad_tensor = pad_tensor.cuda()
    return pad_tensor

def sparse_prefix_pad(inp, sos_idx):
    n_vocab = inp.size(2)
    pad = inp.new_ones(inp.size(0), 1, dtype=torch.long) * sos_idx
    sparse_pad = Tools.one_hot(pad, n_vocab).cuda()
    tensor = torch.cat([sparse_pad, inp], 1)
    return tensor

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder

def kl_loss(prior_dist, posterior_dist):
        bias = 1e-24
        if (len(prior_dist.shape) >= 3) and op.hungary:
            B, S = prior_dist.size(0), prior_dist.size(1)
            expand_prior_dist = prior_dist.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, S * S, -1)
            expand_posterior_dist = posterior_dist.unsqueeze(1).expand(-1, S, -1, -1).reshape(B, S * S, -1)
            cost_vector = F.kl_div((expand_prior_dist + bias).log(), expand_posterior_dist, reduce=False).sum(-1)
            cost_matrix = cost_vector.reshape(-1, S, S)
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            row_idx, col_idx = zip(*[optimize.linear_sum_assignment(cost_matrix_np[i]) for i in range(B)])
            col_idx = torch.tensor(col_idx, dtype=torch.long)  # B, S
            posterior_dist = Tools.nested_index_select(posterior_dist, col_idx)
        flat_prior_dist = prior_dist.reshape(-1, prior_dist.size(-1))
        flat_posterior_dist = posterior_dist.reshape(-1, posterior_dist.size(-1))
        kl_div = F.kl_div((flat_prior_dist + bias).log(), flat_posterior_dist, reduce=False).sum(-1)
        kl_div = kl_div.mean()
        return kl_div

def nll_loss(hypothesis, target, pad_id ):
        eps = 1e-9
        B, T = target.shape
        hypothesis = hypothesis.reshape(-1, hypothesis.size(-1))
        target = target[:,1:]
        padding = torch.ones(target.size(0),1,dtype=torch.long) * pad_id
        padding = padding.cuda()
        target = torch.cat([target,padding],1)
        target = target.reshape(-1)
        nll_loss = F.nll_loss(torch.log(hypothesis + 1e-20), target, ignore_index=pad_id, reduce=False)
        not_ignore_tag = (target != pad_id).float()
        not_ignore_num = not_ignore_tag.reshape(B, T).sum(-1)
        sum_nll_loss = nll_loss.reshape(B, T).sum(-1)
        nll_loss_vector = sum_nll_loss / (not_ignore_num + eps)
        nll_loss = nll_loss_vector.mean()
        return nll_loss, nll_loss_vector.detach()

def regularization_loss(dist):
        entropy_loss, repeat_loss = torch.tensor(0.), torch.tensor(0.)
        if not op.wo_entropy_restrain:
            entropy_loss = Tools.entropy_restrain(dist)
        if not op.wo_repeat_penalty:
            repeat_loss = Tools.repeat_penalty(dist)
        regularization = entropy_loss + repeat_loss
        return regularization

def action_nll(hypothesis,target,pad_idx):
        eps = 1e-9
        hypothesis = hypothesis.reshape(-1,hypothesis.size(-1))
        target = target[:,[1]]
        target = target.reshape(-1)
        nll_loss = F.nll_loss(torch.log(hypothesis+eps),target,ignore_index=pad_idx)
        return nll_loss