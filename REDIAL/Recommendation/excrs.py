import gc
import pickle
import random
import ipdb
from SelfAttention import SelfAttention
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
import torch.nn as nn
import torch
import json
from option import  option as op
from tau_scheduler import  TauScheduler
from copy_scheduler import CopyScheduler
from transformer.Models import Encoder
from transformer.Models import Decoder
from ProfileTopicRedial import PriorProfile,PosteriorProfile
from PreferenceRecRedial import PriorPreference,PosteriorPreference
from gumbel_softmax import GumbelSoftmax
from ActionRecRedial import Action
from tools import Tools
from VocabRedial import Vocab
from scipy import optimize
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformer.Optim import ScheduledOptim
from DataLoaderRecRedial import DataLoaderRec
import math
import sys

class ExcrsTopic(nn.Module):
    def __init__(self,vocab:Vocab,user_cont,n_layers=6,p_layers=3,
                 d_word_vec=512,d_model=512, d_inner=2048,beam_width=1,
                 n_head=8, d_k=64, d_v=64, dropout=0.1):

        super(ExcrsTopic, self).__init__()

        self.vocab = vocab
        self.glo2loc , self.loc2glo = vocab.vocab_transfer()
        self.glo2loc = torch.tensor(self.glo2loc).cuda()
        self.loc2glo = torch.tensor(self.loc2glo).cuda()
        self.word_vocab, self.word_len, self.topic_vocab, self.topic_len= vocab.get_vocab()
        self.word_pad_idx = vocab.get_word_pad()
        self.topic_pad_idx = vocab.get_topic_pad()
        self.kg = pickle.load(open("./dataset/subkg.pkl", "rb"))
        edge_list, n_relation = _edge_list(self.kg, self.topic_len, hop=2)
        edge_list = list(set(edge_list))
        self.dbpedia_edge_sets = torch.LongTensor(edge_list).cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]
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

        self.attn_c = SelfAttention(d_model).cuda()
        self.attn_tp = SelfAttention(d_model).cuda()

        #  encode U_t, context, conversation
        self.main_tfr_encoder = Encoder(
            n_src_vocab=self.word_len, n_position=op.context_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.word_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.word_emb
        ).cuda()

       
        self.u_tfr_encoder4p = Encoder(
            n_src_vocab=user_cont, n_position=1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.user_emb
        ).cuda()

        self.u_tfr_encoder4q = Encoder(
            n_src_vocab=user_cont, n_position=1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.user_emb
        ).cuda()

        # encode graph

        self.dbpedia_RGCN = RGCNConv(d_word_vec, d_word_vec, n_relation, num_bases=8).cuda()

        # for prior profile,preference
        self.p_tfr_encoder4p = Encoder(
            n_src_vocab=self.topic_len, n_position=500,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        ).cuda()

        self.p_tfr_decoder4p = Decoder(
            n_trg_vocab=self.topic_len, n_position=50,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        ).cuda()


        # for posterior profile,preference
        self.p_tfr_encoder4q = Encoder(
            n_src_vocab=self.topic_len, n_position=200,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        ).cuda()
        self.p_tfr_decoder4q = Decoder(
            n_trg_vocab=self.topic_len, n_position=50,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        ).cuda()

       
        self.a_tfr_decoder = Decoder(
            n_trg_vocab=self.topic_len, n_position=op.action_num,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        ).cuda()

              
        if op.wo_l:
            self.p_l = None
            self.q_l = None
        else:
            self.p_l = PriorProfile(encoder=self.u_tfr_encoder4p,decoder=self.p_tfr_decoder4p,
                                    hidden_size=d_model,n_topic_vocab=self.topic_len,
                                    trg_bos_idx=self.l_bos_idx,max_seq_len=op.profile_num,
                                    gs=self.gumbel_softmax,ts=self.pro_tau_scheduler).cuda()

            self.q_l = PosteriorProfile(main_encoder=self.main_tfr_encoder,id_encoder = self.u_tfr_encoder4q,
                                    topic_encoder=self.p_tfr_encoder4q, decoder=self.p_tfr_decoder4q,
                                    hidden_size=d_model,n_topic_vocab=self.topic_len,
                                    trg_bos_idx=self.l_bos_idx,max_seq_len=op.profile_num,
                                    gs=self.gumbel_softmax,ts=self.pro_tau_scheduler,
                                    glo2loc=self.glo2loc,loc2glo=self.loc2glo).cuda()

        # 3.preference
        if op.wo_m:
            self.p_mt = None
            self.q_mt = None
        else:
            self.p_mt = PriorPreference(encoder=self.p_tfr_encoder4p,decoder=self.p_tfr_decoder4p,
                                        main_tfr_encoder=self.main_tfr_encoder,
                                       hidden_size=d_model,n_topic_vocab=self.topic_len,trg_bos_idx=self.m_bos_idx,
                                       max_seq_len=op.preference_num,gs=self.gumbel_softmax,glo2loc=self.glo2loc,
                                       loc2glo=self.loc2glo,ts=self.pre_tau_scheduler).cuda()

            self.q_mt = PosteriorPreference(encoder=self.p_tfr_encoder4q,main_encoder=self.main_tfr_encoder,
                                          decoder=self.p_tfr_decoder4q,
                                         hidden_size=d_model,n_topic_vocab=self.topic_len,trg_bos_idx=self.m_bos_idx,
                                         max_seq_len=op.preference_num,gs=self.gumbel_softmax,glo2loc=self.glo2loc,
                                         loc2glo=self.loc2glo,ts=self.pre_tau_scheduler).cuda()

        
        self.action = Action(p_encoder=self.p_tfr_encoder4p,main_encoder=self.main_tfr_encoder,
                             a_decoder=self.a_tfr_decoder,graphencoder=self.dbpedia_RGCN,hidden_size=d_model,
                             n_topic_vocab=self.topic_len,bos_idx=self.a_bos_idx,vocab=self.vocab,
                             max_len=op.action_num,glo2loc=self.glo2loc,loc2glo=self.loc2glo).cuda()


    def forward(self,
                user_id,
                all_topic, all_topic_len,
                context, context_len,
                tp_path, tp_path_len,
                ar_gth, ar_gth_len,
                related_topic,related_topic_len,
                final,
                pv_m,
                mode='train'):

        assert mode in ['train','valid','test']
        B = op.batch_size
        pv_m, pv_m_mask = self.mask_preference(pv_m, final)
        db_nodes_features = self.dbpedia_RGCN(self.topic_emb.weight, self.db_edge_idx, self.db_edge_type) # [B,L,H]

        tp_path_hidden = self.get_hidden(db_nodes_features,tp_path)
        all_topic_hidden = self.get_hidden(db_nodes_features,all_topic)
        ar_hidden = self.get_hidden(db_nodes_features,ar_gth)
        related_topic_hidden = self.get_hidden(db_nodes_features,related_topic)

        if mode == 'pretrain':
            context_mask = get_mask_via_len(context_len,op.context_max_len)
            context_hidden = self.main_tfr_encoder(context,context_mask)
            context_rep = self.attn_c(context_hidden,context_mask) # [B,1,H]

            tp_path_mask = get_mask_via_len(tp_path_len,op.state_num_redial)
            tp_rep = self.attn_tp(tp_path_hidden,tp_path_mask)   # [B,1,H]

            eps = 1e-9
            hidden_a = context_rep.expand(-1, B, -1)  # B, B, H
            hidden_b = tp_rep.permute(1, 0, 2).expand(B, -1, -1)
            cos_sim_matrix = F.cosine_similarity(hidden_a, hidden_b, dim=-1)  # B, B
            cos_sim_matrix = cos_sim_matrix
            cos_sim_matrix = torch.diag(torch.softmax(cos_sim_matrix, -1))
            loss = - torch.log(cos_sim_matrix + eps).mean()
            return loss

        elif mode == 'train':
            self.global_step += 1

            # user profile
            # p_l, p_l_gumbel = self.p_l.forward(id=user_id)
            # q_l, q_l_gumbel = self.q_l.forward(id=user_id, topics=all_topic, topics_len=all_topic_len,topic_hidden=all_topic_hidden)
            p_l = None
            q_l= None
            q_l_gumbel = None
            # user preference
            p_m, p_m_gumbel = self.p_mt.forward(context=context, context_len=context_len,pv_m=pv_m,pv_m_mask=pv_m_mask,
                                                tp_path=tp_path,tp_path_len=tp_path_len,tp_path_hidden=tp_path_hidden)
            q_m, q_m_gumbel = self.q_mt.forward(context=context, context_len=context_len,pv_m=pv_m,
                                                pv_m_mask=pv_m_mask,ar_gth=ar_gth,ar_gth_len=ar_gth_len,
                                                tp_path=tp_path,tp_path_len=tp_path_len,tp_path_hidden=tp_path_hidden,
                                                ar_hidden=ar_hidden)

            # action
            ar = self.action.forward(m=q_m_gumbel,
                                     l=q_l_gumbel,
                                     context=context,
                                     context_len=context_len,
                                     ar_gth=ar_gth,ar_gth_len=ar_gth_len,
                                     tp_path=tp_path,tp_path_len=tp_path_len,
                                     related_topic=related_topic,related_topic_len=related_topic_len,
                                     tp_path_hidden=tp_path_hidden,related_topic_hidden=related_topic_hidden,
                                     mode='train')

            return p_l, q_l, p_m, q_m, ar,  q_m_gumbel

        else:
            # user profile
            # p_l = self.p_l.forward(id=user_id)


            # user preference
            p_m = self.p_mt.forward(context=context,context_len=context_len,pv_m=pv_m,pv_m_mask=pv_m_mask,
                                    tp_path=tp_path,tp_path_len=tp_path_len,tp_path_hidden=tp_path_hidden)
            p_l = None
            # action
            ar,ar_probs = self.action.forward(m=p_m,
                                     l=p_l,
                                     context=context,
                                     context_len=context_len,
                                     ar_gth=ar_gth,ar_gth_len=ar_gth_len,
                                     tp_path=tp_path,tp_path_len=tp_path_len,
                                     related_topic=related_topic, related_topic_len=related_topic_len,
                                     tp_path_hidden=tp_path_hidden,related_topic_hidden=related_topic_hidden,
                                     mode='test')

            return  ar, ar_probs, p_m, p_l

    def mask_preference(self, pv_m, final):
        #  [B,L,V]   [B,]  
        b = range(op.batch_size)
        b = [i + 1 for i in b]
        # final = final.cpu()
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

    def get_hidden(self,nodes,topics):
        related_topic_hidden = None
        for i in range(op.batch_size):
            t = topics[i]
            if related_topic_hidden is None:
                related_topic_hidden = nodes[t].unsqueeze(0)
            else:
                related_topic_hidden = torch.cat([related_topic_hidden, nodes[t].unsqueeze(0)], 0)
        return related_topic_hidden

class EngineTopic():
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
        self.mask = torch.zeros(op.batch_size, 1, self.model.topic_len).cuda()
        self.movies = []
        movie_id = json.load(open('./dataset/movie_id.json'))
        for movie in movie_id:
            self.movies.append(int(movie))
        self.movie_len = len(self.movies)

    def train(self,train_set,test_set):
        bst_metric = 0
        patience = 0
        gen_stop = False

        # pre-train
        for e in range(3):
            train_loader = DataLoaderRec(train_set, self.vocab)
            self.optimizer.zero_grad()
            for index,input in enumerate(train_loader):
                if input[0].size(0) != op.batch_size:
                    break

                id, \
                context_idx, context_len, \
                state_U, state_U_len, \
                all_topic, all_topic_len, \
                a_R, ar_gth_len, \
                related_topic,related_topic_len,\
                final = input

                a_R = a_R.unsqueeze(1)
                loss = self.model.forward(user_id=id,
                                           all_topic=all_topic, all_topic_len=all_topic_len,
                                           context=context_idx, context_len=context_len,
                                           tp_path=state_U, tp_path_len=state_U_len,
                                           ar_gth=a_R, ar_gth_len=ar_gth_len,
                                           related_topic=related_topic,
                                           related_topic_len=related_topic_len,
                                           final=final,
                                           pv_m=None,
                                           mode='pretrain')

                if (self.global_step % 200 == 0):
                    print("global_step: {}".format(self.global_step))
                    print("kl_preference: {}".format(loss))
                    sys.stdout.flush()

                loss = loss / float(op.gradient_stack)
                loss.backward(retain_graph=False)
                if self.global_step % op.gradient_stack == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.global_step += 1

        for e in range(op.epoch):
            print("epoch : {}".format(e))
            # train_loader = DataLoaderTopic(train_set,self.vocab)
            train_loader = DataLoaderRec(train_set, self.vocab)

            # init preference   [B,L_m,V]
            pv_m = get_default_tensor([op.batch_size, op.preference_num, self.model.topic_len], torch.float,
                                      pad_idx=self.topic_pad_idx)
            self.optimizer.zero_grad()
            for index,input in enumerate(train_loader):
                if input[0].size(0) != op.batch_size:
                    break

                id, \
                context_idx, context_len, \
                state_U, state_U_len, \
                all_topic, all_topic_len, \
                a_R, ar_gth_len, \
                related_topic,related_topic_len,\
                final = input

                a_R = a_R.unsqueeze(1)
                p_l, q_l,  p_m, q_m, ar,  m= self.model.forward(user_id=id,
                                                                        all_topic=all_topic,all_topic_len=all_topic_len,
                                                                        context=context_idx, context_len=context_len,
                                                                        tp_path=state_U,tp_path_len=state_U_len,
                                                                        ar_gth=a_R, ar_gth_len=ar_gth_len,
                                                                        related_topic=related_topic,related_topic_len=related_topic_len,
                                                                        final=final,
                                                                        pv_m=pv_m)


                '''loss'''
                # kl_l = kl_loss(p_l, q_l.detach())
                # self.kl_l_loss += kl_l.item()
                kl_m = kl_loss(p_m, q_m.detach())
                self.kl_m_loss += kl_m.item()
                nll_ar = action_nll(ar, a_R.detach(), self.model.topic_pad_idx)
                self.action_loss += nll_ar.item()

                # p_l_reg, q_l_reg = regularization_loss(p_l), regularization_loss(q_l)
                p_m_reg, q_m_reg = regularization_loss(p_m), regularization_loss(q_m)

                reg_loss = op.reg_lambda * ( p_m_reg + q_m_reg ) #p_l_reg + q_l_reg +

                loss = 0.1 *  kl_m  + nll_ar + reg_loss
                       # +0.1* kl_l

                if (self.global_step % 200 == 0):
                    print("global_step: {}".format(self.global_step))
                    print("kl_preference: {}".format(self.kl_m_loss / self.model.global_step))
                    # print("kl_profile: {}".format(self.kl_l_loss / self.model.global_step))
                    print("nll_ar: {}".format(self.action_loss / self.model.global_step))
                    sys.stdout.flush()

                loss = loss / float(op.gradient_stack)
                loss.backward(retain_graph=False)
                if self.global_step % op.gradient_stack == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.global_step += 1
                pv_m = m.detach()

            self.test(test_set,'test')

            # 释放内存
            del train_loader
            gc.collect()
            if bst_metric > self.metrics["recall@1"]+self.metrics["recall@10"]+self.metrics["recall@50"]:
                patience += 1
                print(f"[Patience = {patience}]")
                if patience >= op.max_patience:
                    gen_stop = True
            else:
                patience = 0
                bst_metric = self.metrics["recall@1"]+self.metrics["recall@10"]+self.metrics["recall@50"]
                if self.metrics["recall@1"]/self.metrics["count"] > 0.04 and self.metrics["recall@50"]/self.metrics["count"] >0.4:
                    torch.save(self.model, './rec_redial.pkl')
                    print("saved:{}".format(self.metrics))
            if gen_stop == True:
                break

        print("train finished ! ")

    def test(self,test_set,mode):

        assert mode in ["test","valid"]

        self.model.eval()
        if mode == "valid":
            print(" valid ")
            dataloader = DataLoaderRec(test_set,self.vocab)
        else:
            print(" test ")
            dataloader = DataLoaderRec(test_set,self.vocab)

        self.metrics = {"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}

        # init preference   [B,L_m,V]
        pv_m = get_default_tensor([op.batch_size, op.preference_num, self.model.topic_len], torch.float,
                                  pad_idx=self.model.topic_pad_idx)

        with torch.no_grad():

            for index,data in enumerate(dataloader):
                if data[0].size(0) != op.batch_size:
                    break

                id, \
                context_idx, context_len, \
                state_U, state_U_len, \
                all_topic, all_topic_len, \
                a_R, ar_gth_len, \
                related_topic, related_topic_len, \
                final  = data


                a_R = a_R.unsqueeze(1)
                ar, ar_probs, m , l = self.model.forward(user_id=id,
                                                    all_topic=all_topic,all_topic_len=all_topic_len,
                                                    context=context_idx, context_len=context_len,
                                                    tp_path=state_U,tp_path_len=state_U_len,
                                                    ar_gth=a_R, ar_gth_len=ar_gth_len,
                                                         related_topic=related_topic,
                                                         related_topic_len=related_topic_len,
                                                    final=final,
                                                    pv_m=pv_m,
                                                     mode='test'
                                                     )

                pv_m = one_hot_scatter(m,self.vocab.topic_num())
                self.compute_metrics(ar_probs, a_R)

        output_dict_rec = {key: self.metrics[key] / self.metrics['count'] for key in self.metrics}
        output_dict_rec['count'] = self.metrics['count']
        print(output_dict_rec)
        self.model.train()
        print('test finished!')
        sys.stdout.flush()
        # 释放内存
        del dataloader
        gc.collect()


    def compute_metrics(self,ar_probs, ar_gth):
        '''
        ar_probs  [B,L,V]
        ar_gth    [B,L]
        '''
        
        ar_probs = ar_probs.squeeze(1)
        _, pred_idx = torch.topk(ar_probs, k=100, dim=1)
        for i in range(op.batch_size): 
            target = ar_gth[i,:]
            self.metrics["recall@1"] += int(target in pred_idx[i][:1].tolist())
            self.metrics["recall@10"] += int(target in pred_idx[i][:10].tolist())
            self.metrics["recall@50"] += int(target in pred_idx[i][:50].tolist())
            self.metrics["count"] += 1


def get_mask_via_len(length, max_len):
    """"""
    B = length.size(0)  # batch size
    mask = torch.ones([B, max_len]).cuda()
    mask = torch.cumsum(mask, 1)  # [ [1,2,3,4,5..], [1,2,3,4,5..] .. ] [B,max_len]
    mask = mask <= length.unsqueeze(
        1) 
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
        target = target.reshape(-1)
        nll_loss = F.nll_loss(torch.log(hypothesis+eps),target,ignore_index=pad_idx)

        return nll_loss

def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)