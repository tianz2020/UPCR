import json
from option import  option
from DataProcessor import clip_pad_sentence,clip_pad_context
import torch
import pandas as pd
import re

class DataLoaderRec():
    def __init__(self,dataset,vocab):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = option.batch_size
        self.history_convs = [ [] for _ in range(self.batch_size)]
        self.number_workers = option.worker_num
        self.sunset=False
        self.conv_index = 0
        self.name2id = self.get_name2id()
        self.topic_graph = self.get_topic_graph()

    def __iter__(self):
        return self

    def __next__(self):
        for i in range(len(self.history_convs)):
            if len(self.history_convs[i]) == 0:
                if not self.sunset:
                    processed_session = self.load_processed_session()
                    if processed_session is not None:
                        self.history_convs[i] = processed_session
        self.history_convs = [ conv for conv in self.history_convs if len(conv)>0 ]
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
        conv = self.dataset[self.conv_index]
        processed_session = self.process(conv)
        self.conv_index += 1
        return processed_session

    def process(self,conversation):
        session_segs = []
        id = int(conversation[0])
        contexts = conversation[-2]
        all_topics = conversation[-1]
        utterances = conversation[1:-2]
        uttr_len = len(utterances)
        all_topic, all_topic_len = clip_pad_sentence(all_topics, option.all_topic_num)
        all_topic = self.vocab.topic2index(all_topic)
        for i in range(2,uttr_len,2):
            response = utterances[i]
            resp = response[0]
            resp, resp_len = clip_pad_sentence(resp, max_len=option.r_max_len, sos=option.BOS_RESPONSE, eos=option.EOS_RESPONSE)
            action_R = response[2]
            if action_R == []:
                continue
            a_R, a_R_len = clip_pad_sentence(action_R, option.action_num)
            context = contexts[:i]
            context, context_len = clip_pad_context(context,option.context_max_len)
            state_R = utterances[i-2][1]
            state_R, state_R_len = clip_pad_sentence(state_R, option.state_num)
            Seeker = utterances[i-1]
            seek = Seeker[0]
            seek, seek_len = clip_pad_sentence(seek, max_len=option.r_max_len, sos=option.BOS_CONTEXT,eos=option.EOS_CONTEXT, save_prefix=False)
            state_U = Seeker[1]
            state_U, state_U_len = clip_pad_sentence(state_U,option.state_num)
            action_U = Seeker[1][-1]
            pv_action = action_U
            related_topics = self.get_related_movies(pv_action,option.movie_num)
            related_topics, related_topics_len = clip_pad_sentence(related_topics,option.movie_num)
            context_idx = self.vocab.word2index(context)
            seek_idx = self.vocab.word2index(seek)
            resp_idx = self.vocab.word2index(resp)
            state_R = self.vocab.topic2index(state_R)
            a_R = self.vocab.topic2index(a_R)
            state_U = self.vocab.topic2index(state_U)
            related_topics = self.vocab.topic2index(related_topics)
            session_segs.append([id,all_topic, all_topic_len,context_idx,context_len,
                                 state_U, state_U_len,a_R, a_R_len,seek_idx,seek_len,
                                 resp_idx,resp_len,state_R,state_R_len,related_topics,related_topics_len,1])
        session_segs[0][-1] = 0
        return session_segs

    def get_topic_graph(self):
        with open('./dataset/graph_rec.json') as f:
            topic_graph = json.load(f)
        return topic_graph

    def get_related_movies(self, action_U, relation_num):
        return self.topic_graph[action_U][0:relation_num]

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
