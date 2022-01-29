import gc
import json

import torch.nn as nn
import torch
from option import  option as op
from tau_scheduler import  TauScheduler
from copy_scheduler import CopyScheduler
from transformer.Models import Encoder
from transformer.Models import Decoder
from ProfileTopic import PriorProfile,PosteriorProfile
from PreferenceTopic import PriorPreference,PosteriorPreference
from gumbel_softmax import GumbelSoftmax
from UserIntention import UserIntention
from ActionTopic import Action
from tools import Tools
from kg.knowledgeGraph import knowledgeGraph
import ipdb
from Response import  Response
from Vocab import Vocab
from gcn import GraphEncoder
from scipy import optimize
import torch.nn.functional as F
from tqdm import tqdm
from transformer.Optim import ScheduledOptim
import Bleu
import distinct
from DataLoaderResp import DataLoaderResp

import math
import sys
from math import exp

# conda activate mywork
# cd /data/tianzhi-slurm/code/mycode/myWork/


class ExcrsResp(nn.Module):
    def __init__(self,vocab:Vocab,user_cont,n_layers=6,p_layers=3,
                 d_word_vec=512,d_model=512, d_inner=2048,beam_width=1,
                 n_head=8, d_k=64, d_v=64, dropout=0.1):

        super(ExcrsResp, self).__init__()

        self.vocab = vocab
        self.glo2loc , self.loc2glo = vocab.vocab_transfer()
        self.glo2loc = torch.tensor(self.glo2loc).cuda()
        self.loc2glo = torch.tensor(self.loc2glo).cuda()
        self.topic_num = vocab.topic_num()
        self.word_vocab, self.word_len, self.topic_vocab, self.topic_len,self.movie_vocab, self.movie_len = vocab.get_vocab()
        self.word_pad_idx = vocab.get_word_pad()
        self.topic_pad_idx = vocab.get_topic_pad()
        self.r_bos_idx = vocab.word2index(op.BOS_RESPONSE)
        self.r_eos_idx = vocab.word2index(op.EOS_RESPONSE)
        self.beam_width = beam_width
        self.word_emb = nn.Embedding(self.word_len,d_word_vec,padding_idx=self.word_pad_idx)
        self.topic_emb = nn.Embedding(self.topic_len,d_word_vec,padding_idx=self.topic_pad_idx)
        self.gumbel_softmax = GumbelSoftmax()
        self.global_step = 0

        #  encode context
        self.main_tfr_encoder = Encoder(
            n_src_vocab=self.word_len, n_position=op.conv_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.word_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.word_emb
        )

        # for encode topic path
        self.p_tfr_encoder4p = Encoder(
            n_src_vocab=self.topic_len, n_position=20,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )

        # encode action
        self.a_tfr_encoder = Encoder(
            n_src_vocab=self.topic_len, n_position=op.action_num,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )

        # decode response
        self.main_tfr_decoder = Decoder(
            n_trg_vocab=self.word_len, n_position=op.r_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.word_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.word_emb
        )

        # Response
        self.response = Response(a_encoder=self.a_tfr_encoder,decoder=self.main_tfr_decoder,p_encoder=self.p_tfr_encoder4p,
                                 main_encoder=self.main_tfr_encoder,hidden_size=d_model,n_vocab=self.word_len,trg_bos_idx=self.r_bos_idx,
                                 trg_eos_idx=self.r_eos_idx,max_seq_len=op.r_max_len,beam_width=beam_width,
                                 loc2glo=self.loc2glo,n_topic=self.topic_len).cuda()


    def forward(self,
                user_id,
                context, context_len,
                tp_path, tp_path_len,
                ar_gth, ar_gth_len,
                resp, resp_len,
                final,
                mode='train'):
        assert mode in ['train','valid','test']

        if mode == 'train':
            self.global_step += 1
            resp = self.response.forward(ar=ar_gth, ar_len=ar_gth_len, context=context,context_len=context_len,
                                         tp_path=tp_path,tp_path_len=tp_path_len,resp_gth=resp, resp_gth_len=resp_len)
            return resp

        else:
            resp,probs = self.response.forward(ar=ar_gth, ar_len=ar_gth_len, context=context, context_len=context_len,
                                         tp_path=tp_path, tp_path_len=tp_path_len, resp_gth=resp, resp_gth_len=resp_len)
            return resp,probs


    def topictensor2nl(self,tensor):
        words = tensor.detach().cpu().numpy()
        words = self.vocab.index2topic(words)
        return words


class EngineResp():
    def __init__(self,model:torch.nn.Module,
                 vocab):
        self.model = model
        lr = op.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = ScheduledOptim(self.optimizer, 0.5, op.d_model, op.n_warmup_steps)
        self.vocab = vocab
        self.topic_pad_idx = self.vocab.topic2index(op.PAD_WORD)
        self.word_pad_idx = self.vocab.word2index(op.PAD_WORD)
        self.global_step = 0
        self.loss = 0
        self.topic_model = torch.load('./topic_graph.pkl')
    def train(self,train_set,test_set):

        for e in range(op.epoch):
            print("epoch : {}".format(e))
            train_loader = DataLoaderResp(train_set,self.vocab)

            self.optimizer.zero_grad()
            for index,input in enumerate(train_loader):
                if input[0].size(0) != op.batch_size:
                    break

                id, \
                context_idx, context_len, \
                state_U, state_U_len, \
                a_R, a_R_len, \
                resp, resp_len, \
                final = input


                resp_gen = self.model.forward(user_id=id,
                                        context=context_idx, context_len=context_len,
                                        tp_path=state_U,tp_path_len=state_U_len,
                                        ar_gth=a_R, ar_gth_len=a_R_len,
                                        resp=resp,resp_len=resp_len,
                                        final=final)

                '''loss'''
                loss,_ = nll_loss(resp_gen,resp.detach(),self.word_pad_idx)

                self.loss += loss.item()
                if (self.global_step % 200 == 0):
                    print("global_step: {}".format(self.global_step))
                    print("loss: {}".format(self.loss / self.model.global_step))
                    sys.stdout.flush()

                loss = loss / float(op.gradient_stack)
                loss.backward(retain_graph=False)
                if self.global_step % op.gradient_stack == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.global_step += 1

            self.test(test_set,'test')

            # 释放内存
            del train_loader
            gc.collect()
        # torch.save(self.model,'./model_topic.pkl')
        print("train finished ! ")

    def test(self,test_set,mode):
        assert mode in ["test","valid"]
        res_gen = []
        res_gth = []
        losses = 0
        step = 0
        self.model.eval()
        if mode == "valid":
            print(" valid ")
            dataloader = DataLoaderResp(test_set,self.vocab)
        else:
            print(" test ")
            dataloader = DataLoaderResp(test_set,self.vocab)
            # dataloader = json.load(open('./dataset/test_resp.json'))

        with torch.no_grad():
            # for index,data in dataloader.items():
            for index,data in enumerate(dataloader):
                if data[0].size(0) != op.batch_size:
                    break

                step += 1

                id, \
                context_idx, context_len, \
                state_U, state_U_len, \
                a_R, a_R_len, \
                resp, resp_len, \
                final = data

                # context_idx = torch.tensor(data['context']).cuda()
                # context_len = torch.tensor(data['context_len']).cuda()
                # state_U = torch.tensor(data['state_U']).cuda()
                # state_U_len = torch.tensor(data['state_U_len']).cuda()
                # a_R = torch.tensor(data['ar']).cuda()
                # a_R = a_R.unsqueeze(0)
                # a_R_len = torch.tensor(data['ar_len']).cuda()
                # resp = torch.tensor(data['resp']).cuda()

                # resp_gen,probs = self.model.forward(user_id=None,
                #                             context=context_idx, context_len=context_len,
                #                             tp_path=state_U,tp_path_len=state_U_len,
                #                             ar_gth=a_R, ar_gth_len=a_R_len,
                #                             resp=None,resp_len=None,
                #                             final=None,
                #                             mode='test')

                probs = self.model.forward(user_id=id,
                                        context=context_idx, context_len=context_len,
                                        tp_path=state_U,tp_path_len=state_U_len,
                                        ar_gth=a_R, ar_gth_len=a_R_len,
                                        resp=resp,resp_len=resp_len,
                                        final=final)

                loss, _ = nll_loss(probs, resp.detach(), self.word_pad_idx)
                losses += loss.item()

            ppl = exp(losses/step)
            print("ppl:{}".format(ppl))
            #     resp_gen_word = self.wordtensor2nl(resp_gen)
            #     resp_gth_word = self.wordtensor2nl(resp)
            #
            #     res_gen.extend(resp_gen_word)
            #     res_gth.extend(resp_gth_word)
            #
            # bleu_1, bleu_2, bleu_3, bleu_4 = Bleu.bleu(res_gen,res_gth)
            # dist_1, dist_2 = distinct.cal_calculate(res_gen,res_gth)
            # print("bleu_1:{},bleu_2:{},bleu_3:{},bleu_4:{},dist_1:{},dist_2:{}".format(bleu_1,bleu_2,bleu_3,bleu_4,dist_1,dist_2))
            sys.stdout.flush()
        self.model.train()
        print('test finished!')
        # 释放内存
        del dataloader
        gc.collect()

    def wordtensor2nl(self,tensor):
        words = tensor.detach().cpu().numpy()
        words = self.vocab.index2word(words)
        return words
    def compute_metrics(self,ar_probs, ar_gth, a_R_len, metrics):
        '''
        ar_probs  [B,L,V]
        ar_gth    [B,L]
        '''
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
            metrics['rec_count'] += 1
            for k in [1,10,50]:
                pred, pred_id = torch.topk(gen,k,-1)
                pred_id = pred_id.tolist()
                if tar in pred_id:
                    rank = pred_id.index(tar)
                    metrics['NDCG{}'.format(k)] += 1.0 / math.log(rank + 2.0, 2)
                    metrics['MRR{}'.format(k)] += 1.0 / (rank + 1.0)

        for i, gt in enumerate(ar_gth):  # 循环每个batch
            # gt是一个batch的action  [type,topic,type,topic...]
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
    """"""
    B = length.size(0)  # batch size
    mask = torch.ones([B, max_len]).cuda()
    mask = torch.cumsum(mask, 1)  # [ [1,2,3,4,5..], [1,2,3,4,5..] .. ] [B,max_len]
    mask = mask <= length.unsqueeze(
        1)  # [ [True,True,..,Flase],[True,True,..,Flase],..  ] 第一个列表中True的个数为第一个session中的句子长度，后面填充的都是false
    mask = mask.unsqueeze(-2)  # [B,1,max_len]
    return mask

def get_default_tensor(shape, dtype, pad_idx=None):
    pad_tensor = torch.zeros(shape, dtype=dtype)
    pad_tensor[..., pad_idx] = 1.0 if dtype == torch.float else 1
    pad_tensor = pad_tensor.cuda()
    return pad_tensor

def sparse_prefix_pad(inp, sos_idx):
    # tensor [B,Ls,V]

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
        """kl loss, 计算的是posterior和prior kl divergence
        Parameters
        ------
        prior_dist:             B, S, K
        posterior_dist:         B, S, K
        """
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
        """nll  loss
        Parameters
        ------
        hypothesis:         B, T, V
        target:             B, T
        pad_id:             bool

        Returns
        ------
        nll_loss:           (,)   (Scalar)
        nll_loss_vector:    B,
        """

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
        '''
        hypothesis : [B,L,V]
        target  : [B,L]
        '''
        eps = 1e-9
        hypothesis = hypothesis.reshape(-1,hypothesis.size(-1))
        target = target[:,[1,3,5,7,9]]
        target = target.reshape(-1)
        nll_loss = F.nll_loss(torch.log(hypothesis+eps),target,ignore_index=pad_idx)

        return nll_loss