import json
from multiprocessing import Process, Queue
import csv
import ipdb
from tqdm import tqdm
from option import  option
from enum import Enum
from DataProcessor import clip_pad_sentence,clip_pad_context
from Vocab import Vocab
import torch
import pickle
from transformers import BertTokenizer
import json as js

class DataLoaderTopicPretrain():
    def __init__(self,dataset,vocab,task_queue_max_size=1000,
                 processed_queue_max_size=1000):

        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = option.batch_size
        # max_len: response,context,profile,topic
        self.history_convs = [ [] for _ in range(self.batch_size)]

        # self.task_queue = Queue(maxsize=task_queue_max_size)
        # self.processed_queue = Queue(maxsize=processed_queue_max_size)
        self.number_workers = option.worker_num
        self.sunset=False
        self.conv_index = 0
        # self.user2profile = self.get_profile()
        # self.graph = self.get_graph()
        # self.adj = self.get_adj()
        self.topic_graph = self.get_topic_graph()
        self.num = 0
        self.tokenizer = BertTokenizer(vocab_file='./dataset/vocab.txt')

    def __iter__(self):
        # __iter__方法返回一个迭代对象，然后python不断调用该迭代对象的__next__方法拿到循环的下一个值，直到遇到StopIteration退出循环
        return self

    def __next__(self):
        # __iter__方法返回一个迭代对象，然后python不断调用该迭代对象的__next__方法拿到循环的下一个值，直到遇到StopIteration退出循环
        for i in range(len(self.history_convs)):
            if len(self.history_convs[i]) == 0:
                if not self.sunset:
                    processed_session = self.load_processed_session()
                    if processed_session is not None:
                        self.history_convs[i] = processed_session

        self.history_convs = [ conv for conv in self.history_convs if len(conv)>0 ]  # 去除已经完成的
        if len(self.history_convs) == 0:
            print("stop")
            raise StopIteration

        batch_convs = [ conv[0] for conv in self.history_convs ]
        self.history_convs = [ conv[1:] for conv in self.history_convs ]
        nn_inputs = []

        for idx, batch_data in enumerate(zip(*batch_convs)):
            nn_inputs.append(torch.tensor(data=batch_data, dtype=torch.long).cuda())

        return nn_inputs

    def load_processed_session(self):
        if self.conv_index >= len(self.dataset):
            self.sunset = True
            return None

        conv = self.dataset[self.conv_index] #拿出一个session
        processed_session = self.process(conv)
        self.conv_index += 1
        return processed_session

    def process(self,conversation):
        '''
        process a conversation.

        the input of the t-th turn:
        [
            Uid
            profile

            [ [sentence tokens],[state],[action],[turn] ]
            [ [sentence tokens],[state],[action],[turn] ]
            ...

            [ [Rec1],[Seeker1],[Rec2],[Seeker2],... ]    # full conversation
        ]

        the output of the t-th turn :
             element
        [
            Uid                       use for profile prediction
            profile                   待用

            R_{t-1},U_{t}             h_{t}
            len(R_{t-1},U_{t})

            U_{t}                     use for intention prediction
            len(U_{t})

            R_{t}                     use for posterior action prediction
            len(R_{t})

            state_R_{t-1}             use for test,add action_U_{t} to get state_t
            len(state_R_{t-1})

            action_U_{t}              ground truth Seeker intention
            len(action_U_{t})

            state_U_{t}               use for action prediction
            len(state_U_{t})

            action_R_{t}              ground truth Rec action
            len(action_R_{t})

            turn  t                   use  for  action prediction
            all turns                 use for posterior profile prediction
            final_turn                whether it is final turn to get posterior profile
        ]

        '''
        session_segs = []

        id = int(conversation[0])

        all_topics = conversation[-1]
        all_topic, all_topic_len = clip_pad_sentence(all_topics, option.all_topic_num)
        all_topic = self.vocab.topic2index(all_topic)

        length = conversation[-2]
        word2tokens = conversation[-3]
        contexts_token = conversation[-4]
        contexts_word = conversation[-5]

        utterances = conversation[1:-5]
        uttr_len = len(utterances)

        pv_action = []

        for i in range(2,uttr_len,2):
            Seeker = utterances[i - 1]
            # action
            response = utterances[i]  # R_t
            action_R = response[1]

            if action_R == []:
                continue

            # token level
            context_token = contexts_token[:i]

            context_token, context_token_len, word_num = clip_pad_context(context_token,option.context_max_len,length)  # 实际取的token数量
            if context_token_len < option.context_max_len:
                context_token = context_token + [option.PAD_WORD] * (option.context_max_len - context_token_len )

            word_len = len(word_num)   # word_level 实际取的word数量

            # word len
            word_num = word_num + [1] * (option.context_max_len - word_len )  # sequence of word length

            # word_level
            context_word = contexts_word[:i]
            context_word, context_word_len, _ = clip_pad_context(context_word, word_len,length)
            context_word_len = len(context_word)
            if context_word_len < option.context_max_len:
                context_word = context_word + [option.PAD_WORD] * (option.context_max_len - context_word_len )

            # state U_t
            state_U = response[0]
            state_U, state_U_len = clip_pad_sentence(state_U, option.state_num)

            # related_topics
            action_U = Seeker[1]
            if action_U != []:
                pv_action = action_U
            related_topics = self.get_related_topics(pv_action,option.relation_num,action_R)
            related_topics, related_topics_len = clip_pad_sentence(related_topics,option.relation_num)

            # convert token to index
            context_word_idx = self.vocab.word2index(context_word)
            context_token_idx = self.tokenizer.convert_tokens_to_ids(context_token)
            state_U = self.vocab.topic2index(state_U)
            related_topics = self.vocab.topic2index(related_topics)
            a_R, a_R_len = clip_pad_sentence(action_R,option.action_num)
            a_R = self.vocab.topic2index(a_R)

            session_segs.append([id,
                                 context_word_idx,context_word_len,  # B,L,H
                                 context_token_idx,context_token_len,
                                 word_num,   # [B,L,H]
                                 state_U,state_U_len,
                                 related_topics, related_topics_len,
                                 a_R,a_R_len,
                                 all_topic,all_topic_len,
                                 1])
        session_segs[0][-1] = 0
        return session_segs

    def get_topic_graph(self):
        with open('./dataset/topic_graph_1124.json') as f:
            topic_graph = json.load(f)
        f.close()
        return topic_graph


    def get_related_topics(self, action_U, relation_num,action_R):
        gth = []
        for i in range(0,len(action_R),2):
            gth.append(action_R[i+1])
        related_topics = []
        a_len = len(action_U)
        for k in range(0,a_len,2):
            action_type = action_U[k]
            topic = action_U[k+1]
            if '拒绝' in action_type:
                assert a_len > 1
                continue
            related_topic = self.topic_graph[topic][0:int(2*relation_num/a_len)]
            # for t in gth:
            #     if t not in related_topic :
            #         related_topic = [t] + related_topic[:-1]
                    # self.topic_graph[topic].remove(t)
                    # self.topic_graph[topic] = [t] + self.topic_graph[topic]
            related_topics.extend(related_topic)
        return related_topics

    def get_word_len(self, length, contexts_token_len):
        '''
        length : [ 1,1,2,3,1,1 ]
        contexts_token_len = ..
        '''
        token_len = 0
        word_len = 0
        ln = length[:contexts_token_len-2]

        for l in ln:
            token_len += l
            if token_len > contexts_token_len-2:
                return word_len
            word_len+=1

        return word_len

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder

# class TaskAllocator(Process):
#     def __init__(self, task_queue: Queue, conversations, worker_num, show_process_bar=False):
#         """TaskAllocator: put session to task_queue in its whole lifecycle"""
#         super(TaskAllocator, self).__init__()
#         self.task_queue = task_queue
#         self.conversations = conversations
#         self.worker_num = worker_num
#         self.show_process_bar = show_process_bar
#
#
#     def run(self) -> None:
#         if self.show_process_bar:
#             pbar = tqdm(self.conversations)
#         else:
#             pbar = self.conversations
#
#         for idx, session in enumerate(pbar):
#             self.task_queue.put(MetaInfo(MetaType.UNFINISHED, session))
#
#         for _ in range(self.worker_num):
#             # finish tag
#             self.task_queue.put(MetaInfo(MetaType.FINISHED, None))
#
#
# class TaskWorker(Process):
#     def __init__(self, vocab, queue: Queue, output_queue: Queue
#                  , supervised=False):
#         """TaskWorker: works continuously until receive a FINISHED tag"""
#         super(TaskWorker, self).__init__()
#         self.vocab  = vocab
#         self.input_queue = queue
#         self.output_queue = output_queue
#         self.supervised = supervised
#         self.action_types = ['谈论','请求推荐','推荐电影']
#
#     def run(self) -> None:
#
#         while True:
#             meta_info = self.input_queue.get()
#
#             if meta_info.meta_type == MetaType.FINISHED:
#                 break
#
#             session = meta_info.info
#
#             processed_session = self.process(session)
#
#             if len(processed_session) > 0:
#                 self.output_queue.put(processed_session)
#
#
#     def process(self,conversation):
#         '''
#         process a conversation.
#
#         the input of the t-th turn:
#         [
#             Uid
#             profile
#
#             [ [sentence tokens],[state],[action],[turn] ]
#             [ [sentence tokens],[state],[action],[turn] ]
#             ...
#
#             [ [Rec1],[Seeker1],[Rec2],[Seeker2],... ]    # full conversation
#         ]
#
#         the output of the t-th turn :
#              element
#         [
#             Uid                       use for profile prediction
#             profile                   待用
#
#             R_{t-1},U_{t}             h_{t}
#             len(R_{t-1},U_{t})
#
#             U_{t}                     use for intention prediction
#             len(U_{t})
#
#             R_{t}                     use for posterior action prediction
#             len(R_{t})
#
#             state_R_{t-1}             use for test,add action_U_{t} to get state_t
#             len(state_R_{t-1})
#
#             action_U_{t}              ground truth Seeker intention
#             len(action_U_{t})
#
#             state_U_{t}               use for action prediction
#             len(state_U_{t})
#
#             action_R_{t}              ground truth Rec action
#             len(action_R_{t})
#
#             turn  t                   use  for  action prediction
#             all turns                 use for posterior profile prediction
#             final_turn                whether it is final turn to get posterior profile
#         ]
#
#         '''
#
#         session_segs = []
#         history = []
#
#
#         id = int(conversation[0])
#
#         # profile
#         profile = conversation[1]
#         profile_len = len(profile)
#         if len(profile) > option.profile_num:
#             profile = profile[-option.profile_num:]
#         else:
#             profile = profile + [option.PAD_WORD] * (option.profile_num - len(profile))
#         profile = self.vocab.topic2index(profile)
#
#         # conv
#         all_turn = conversation[-1]
#         tmp = []
#         tmp.append(option.BOS_CONTEXT)
#         for turn in all_turn[:-1]:
#             turn = turn + [option.SENTENCE_SPLITER]
#             tmp = tmp + turn
#         tmp = tmp + all_turn[-1]
#         tmp.append(option.EOS_CONTEXT)
#         all_turn = tmp
#         conv_len = len(all_turn)
#         if len(all_turn) < option.conv_max_len:
#             all_turn = all_turn + [option.PAD_WORD] * (option.conv_max_len - len(all_turn))
#         else:
#             all_turn = all_turn[-option.conv_max_len:]
#         all_turn = self.vocab.word2index(all_turn)
#
#
#         utterances = conversation[2:-1]
#         uttr_len = len(utterances)
#
#         for i in range(2,uttr_len,2):
#             pv_r_u = [x[0] for x in utterances[max(0,i-2):i]] # R_{t-1},U_{t}
#             history = history + pv_r_u
#
#             # history=[r_{t-1},sensplit,u_{t}]
#             tmp = []
#             for idx, sentence in enumerate(history[:-1]):
#                 sentence = sentence + [option.SENTENCE_SPLITER]
#                 tmp = tmp + sentence
#             tmp += history[-1]
#             history = tmp
#
#             # state R_{t-1}
#             state_R = utterances[i-2][1]
#             state_R_len = len(state_R)
#             if len(state_R) < option.state_num:
#                 state_R = state_R + [option.PAD_WORD] * (option.state_num - len(state_R))
#             else:
#                 state_R = state_R[-option.state_num:]
#
#             Seeker = utterances[i-1]  # U_{t}
#             seek = Seeker[0]  # U_{t} content
#             # state U_{t}
#             state_U = Seeker[1]
#             state_U_len = len(state_U)
#             if len(state_U) < option.state_num:
#                 state_U = state_U + [option.PAD_WORD] * (option.state_num - len(state_U))
#             else:
#                 state_U = state_U[-option.state_num:]
#
#             action_U = Seeker[2]
#             # a_U_len = len(action_U)
#             a_U = []
#             a_U.append(option.BOS_ACTION)
#
#             # for j in range(0,a_U_len,2) :
#             #     type = action_U[j]
#             #     topic = action_U[j+1]
#             #     a_U.append(type)
#             #     a_U.extend(topic)
#             #     a_U.append(option.ACTION_SPLITER)
#             # a_U = a_U[:-1]
#             a_U.extend(action_U)
#             a_U.append(option.EOS_ACTION)
#             a_U_len = len(a_U)
#             a_U = a_U + [option.PAD_WORD] * (option.action_num - len(a_U))
#
#             response = utterances[i] # R_{t}
#             resp = response[0] # R_{t} content
#
#             action_R = response[2]
#             # a_R_len = len(action_R)
#             a_R = []
#             a_R.append(option.BOS_ACTION)
#             # for j in range(0, a_R_len, 2):
#             #     type = action_R[j]
#             #     topic = action_R[j + 1]
#             #     a_R.append(type)
#             #     a_R.extend(topic)
#             #     a_R.append(option.ACTION_SPLITER)
#             # a_R = a_R[:-1]
#             a_R.extend(action_R)
#             a_R.append(option.EOS_ACTION)
#             a_R_len = len(a_R)
#             a_R = a_R + [option.PAD_WORD] * (option.action_num - len(a_R))
#
#             # history = history + action_types
#             # ['我最近还不错啦，土豪金下周到手，是公司奖励。', 'SENTENCE_SPLITER', '真是优秀啊！我不太在意奖励的，奖励就先不提了，还是脚踏实地认真工作吧，做个努力上进的好孩子吧。', ['谈论']]
#
#             history , history_len = clip_pad_sentence(history,max_len=option.context_max_len,sos=option.BOS_CONTEXT,
#                                                       eos=option.EOS_CONTEXT,save_prefix=False)
#             # [bos,R_{t-1},sent,U_{t},eos,pad,pad,...]
#
#             resp , resp_len = clip_pad_sentence(resp,max_len=option.r_max_len,sos=option.BOS_RESPONSE
#                                                 ,eos=option.EOS_RESPONSE)
#             # [bos,R_{t},eos,pad,pad,... ]
#
#             seek , seek_len = clip_pad_sentence(seek,max_len=option.r_max_len,sos=option.BOS_CONTEXT,
#                                      eos=option.EOS_CONTEXT,save_prefix=False)
#
#             # [bos,U_{t},eos,pad,pad,..]
#
#
#             # R_{t-1}U_{t}
#             history_idx = self.vocab.word2index(history)
#             # U_{t}
#             seek_idx = self.vocab.word2index(seek)
#             # R_{t}
#             resp_idx = self.vocab.word2index(resp)
#
#             # action U_{t}
#             a_U = self.vocab.topic2index(a_U)
#             # state U_{t}
#             state_U = self.vocab.topic2index(state_U)
#             # state R_{t-1}
#             state_R = self.vocab.topic2index(state_R)
#             # action R_{t}
#             a_R = self.vocab.topic2index(a_R)
#
#             session_segs.append([id,profile,profile_len,history_idx,history_len,seek_idx,seek_len,resp_idx,resp_len,
#                                  state_R,state_R_len,a_U,a_U_len,state_U,state_U_len,a_R,a_R_len,i,all_turn,conv_len,0])
#             history = []
#         session_segs[-1][-1] = 1
#         return session_segs
#
# class MetaType(Enum):
#     FINISHED = 1
#     UNFINISHED = 2
#
#
#
# class MetaInfo:
#     def __init__(self, meta_type: MetaType, info=None):
#         self.meta_type = meta_type
#         self.info = info