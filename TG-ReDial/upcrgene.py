import torch.nn as nn
import torch
from option import  option as op
from transformer.Models import Encoder
from transformer.Models import Decoder
from gumbel_softmax import GumbelSoftmax
from tools import Tools
from Response import  Response
from Vocab import Vocab
import torch.nn.functional as F
from transformer.Optim import ScheduledOptim
import Bleu
import distinct
from DataLoaderResp import DataLoaderResp
import sys

class Upcrgene(nn.Module):
    def __init__(self,vocab:Vocab,user_cont,n_layers=6,p_layers=3,
                 d_word_vec=512,d_model=512, d_inner=2048,beam_width=1,
                 n_head=8, d_k=64, d_v=64, dropout=0.1):
        super(Upcrgene, self).__init__()
        self.vocab = vocab
        self.glo2loc , self.loc2glo = vocab.vocab_transfer()
        self.glo2loc = torch.tensor(self.glo2loc).cuda()
        self.loc2glo = torch.tensor(self.loc2glo).cuda()
        self.topic_num = vocab.topic_num()
        self.word_vocab, self.word_len, self.topic_vocab, self.topic_len = vocab.get_vocab(task='gene')
        self.word_pad_idx = vocab.get_word_pad()
        self.topic_pad_idx = vocab.get_topic_pad()
        self.r_bos_idx = vocab.word2index(op.BOS_RESPONSE)
        self.r_eos_idx = vocab.word2index(op.EOS_RESPONSE)
        self.beam_width = beam_width
        self.word_emb = nn.Embedding(self.word_len,d_word_vec,padding_idx=self.word_pad_idx)
        self.topic_emb = nn.Embedding(self.topic_len,d_word_vec,padding_idx=self.topic_pad_idx)
        self.gumbel_softmax = GumbelSoftmax()
        self.global_step = 0
        self.main_tfr_encoder = Encoder(
            n_src_vocab=self.word_len, n_position=op.conv_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.word_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.word_emb
        )
        self.p_tfr_encoder4p = Encoder(
            n_src_vocab=self.topic_len, n_position=20,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )
        self.a_tfr_encoder = Encoder(
            n_src_vocab=self.topic_len, n_position=op.action_num,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=p_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.topic_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.topic_emb
        )
        self.main_tfr_decoder = Decoder(
            n_trg_vocab=self.word_len, n_position=op.r_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.word_pad_idx, dropout=dropout, scale_emb=False,
            word_emb=self.word_emb
        )
        self.response = Response(a_encoder=self.a_tfr_encoder,decoder=self.main_tfr_decoder,p_encoder=self.p_tfr_encoder4p,
                                 main_encoder=self.main_tfr_encoder,hidden_size=d_model,n_vocab=self.word_len,trg_bos_idx=self.r_bos_idx,
                                 trg_eos_idx=self.r_eos_idx,max_seq_len=op.r_max_len,beam_width=beam_width,
                                 loc2glo=self.loc2glo,n_topic=self.topic_len).cuda()

    def forward(self,context, context_len,tp_path, tp_path_len,ar_gth, ar_gth_len,
                resp, resp_len,mode='train'):
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

class Engine():
    def __init__(self,model:torch.nn.Module,vocab):
        self.model = model
        lr = op.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = ScheduledOptim(self.optimizer, 0.5, op.d_model, op.n_warmup_steps)
        self.vocab = vocab
        self.topic_pad_idx = self.vocab.topic2index(op.PAD_WORD)
        self.word_pad_idx = self.vocab.word2index(op.PAD_WORD)
        self.global_step = 0
        self.loss = 0

    def train(self,train_set,test_set):
        for e in range(op.epoch):
            print("epoch : {}".format(e))
            train_loader = DataLoaderResp(train_set,self.vocab)
            self.optimizer.zero_grad()
            for index,input in enumerate(train_loader):
                if input[0].size(0) != op.batch_size:
                    break
                id, context_idx, context_len, state_U, state_U_len, a_R, a_R_len, resp, resp_len, final = input
                resp_gen = self.model.forward(context=context_idx, context_len=context_len,tp_path=state_U,tp_path_len=state_U_len,
                                        ar_gth=a_R, ar_gth_len=a_R_len,resp=resp,resp_len=resp_len)
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
            self.test(test_set)
        print("train finished ! ")

    def test(self,test_set):
        print(" test ")
        self.model.eval()
        res_gen = []
        res_gth = []
        step = 0
        dataloader = DataLoaderResp(test_set,self.vocab)
        with torch.no_grad():
            for index,data in enumerate(dataloader):
                if data[0].size(0) != op.batch_size:
                    break
                step += 1
                id, context_idx, context_len, state_U, state_U_len, a_R, a_R_len, resp, resp_len, final = data
                resp_gen,probs = self.model.forward(context=context_idx, context_len=context_len,tp_path=state_U,tp_path_len=state_U_len,
                                                    ar_gth=a_R, ar_gth_len=a_R_len,resp=None,resp_len=None, mode='test')
                resp_gen_word = self.wordtensor2nl(resp_gen)
                resp_gth_word = self.wordtensor2nl(resp)
                res_gen.extend(resp_gen_word)
                res_gth.extend(resp_gth_word)
            bleu_1, bleu_2, bleu_3, bleu_4 = Bleu.bleu(res_gen,res_gth)
            dist_1, dist_2 = distinct.cal_calculate(res_gen,res_gth)
            print("bleu_1:{},bleu_2:{},bleu_3:{},bleu_4:{},dist_1:{},dist_2:{}".format(bleu_1,bleu_2,bleu_3,bleu_4,dist_1,dist_2))
            sys.stdout.flush()
        self.model.train()
        print('test finished!')

    def wordtensor2nl(self,tensor):
        words = tensor.detach().cpu().numpy()
        words = self.vocab.index2word(words)
        return words


def get_mask_via_len(length, max_len):
    """"""
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