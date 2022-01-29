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
import json as js
import pandas as pd
import re
import gc



class DataLoaderRec():
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
        self.name2id = self.get_name2id()
        self.movies2topic = js.load(open('./dataset/movieid2topics.json'))
        self.topic_graph = self.get_topic_graph()
        self.movie1 = []
        self.num = 0
        self.num_1 = 0
        self.graph = json.load(open('./dataset/topic2movie.json'))
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
        contexts = conversation[-2]
        all_topics = conversation[-1]

        # conv, conv_len = clip_pad_context(contexts,option.conv_max_len)
        # conv_idx = self.vocab.word2index(conv)

        utterances = conversation[1:-2]
        uttr_len = len(utterances)

        # profile的后验的条件
        all_topic, all_topic_len = clip_pad_sentence(all_topics, option.all_topic_num)
        all_topic = self.vocab.topic2index(all_topic)


        for i in range(2,uttr_len,2):
            # R_t
            response = utterances[i]  # R_t
            resp = response[0]  # R_t content
            resp, resp_len = clip_pad_sentence(resp, max_len=option.r_max_len, sos=option.BOS_RESPONSE
                                               , eos=option.EOS_RESPONSE)
            # action
            action_R = response[2]
            if action_R == []:
                continue

            a_R, a_R_len = clip_pad_sentence(action_R, option.action_num)

            # R_0U_1...R_{t-1}U_t
            context = contexts[:i]
            context, context_len = clip_pad_context(context,option.context_max_len)

            # state R_{t-1}
            state_R = utterances[i-2][1]
            state_R, state_R_len = clip_pad_sentence(state_R, option.state_num)

            # U_t
            Seeker = utterances[i-1]
            seek = Seeker[0]
            seek, seek_len = clip_pad_sentence(seek, max_len=option.r_max_len, sos=option.BOS_CONTEXT,
                                               eos=option.EOS_CONTEXT, save_prefix=False)

            # state U_t
            state_U = Seeker[1]
            # topic_path,movie_path = self.process_state(state_U)
            state_U, state_U_len = clip_pad_sentence(state_U,option.state_num)


            # intention
            action_U = Seeker[1][-1]
            pv_action = action_U

            # related_topics
            related_topics = self.get_related_movies(pv_action,option.movie_num,action_R)
            # if action_R[1] not in related_topics:
            #     related_topics = [action_R[1]] + related_topics[:-1]
            related_topics, related_topics_len = clip_pad_sentence(related_topics,option.movie_num)

            # convert token to index

            context_idx = self.vocab.word2index(context)
            seek_idx = self.vocab.word2index(seek)
            resp_idx = self.vocab.word2index(resp)

            # topic_path = self.vocab.topic2index(topic_path)
            # movie_path = self.vocab.topic2index(movie_path)

            state_R = self.vocab.topic2index(state_R)
            a_R = self.vocab.topic2index(a_R)
            state_U = self.vocab.topic2index(state_U)
            related_topics = self.vocab.topic2index(related_topics)

            session_segs.append([id,
                                 all_topic, all_topic_len,
                                 context_idx,context_len,
                                 #  这里把topic path改成state_U 了
                                 state_U, state_U_len,
                                 a_R, a_R_len,
                                 seek_idx,seek_len,
                                 resp_idx,resp_len,
                                 state_R,state_R_len,
                                 related_topics,related_topics_len,
                                 1])
        session_segs[0][-1] = 0
        # del utterances
        # del conversation
        # gc.collect()
        return session_segs

    def get_topic_graph(self):
        with open('./dataset/graph_rec_full_1.json') as f:
            topic_graph = json.load(f)
        f.close()
        return topic_graph


    def get_related_movies(self, action_U, relation_num , action):

        related_topic = self.topic_graph[action_U][0:relation_num]
        movie = action[1]
        if movie not in related_topic :
            self.num+=1
        else:
            self.num_1+=1
        # if action_U not in self.graph:
        #     self.graph[action_U] = [movie]
        # else:
        #     if movie not in self.graph[action_U]:
        #         self.graph[action_U] = [movie] + self.graph[action_U]
        return related_topic

    def process_state(self, state_U):
        topic_path = []
        movie_path = []
        for state in state_U:
            if state in self.movies2topic:
                movie_path.append(state)
            elif state in self.topic_graph:
                topic_path.append(state)
            else:
                if state in self.name2id:
                    movie_path.append(self.name2id[state])
        return topic_path, movie_path

    def get_name2id(self):
        name2id = {}
        movie_id = pd.read_csv('./dataset/movie_with_mentions.csv', usecols=[1, 2, 3], encoding='gbk')
        movies = movie_id.values.tolist()
        for movie in movies:
            movie[1] = re.sub('\(\d*\)', '', movie[1])
            movie[1] = re.sub('\(上\)', '', movie[1])
            movie[1] = re.sub('\(下\)', '', movie[1])
            movie[1] = re.sub('\(美版\)', '', movie[1])
            name2id[movie[1]] = str(movie[0])
        return name2id

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