import json
import csv
import ipdb
from tqdm import tqdm
from option import  option
from enum import Enum
from DataProcessor_Redial import clip_pad_sentence,clip_pad_context
from VocabRedial import Vocab
import torch
import pickle
import json as js
import pandas as pd
import re
import gc
import random
from random import  shuffle

class DataLoaderRec():
    def __init__(self,dataset,vocab,task_queue_max_size=1000,
                 processed_queue_max_size=1000):

        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = option.batch_size
        self.history_convs = [ [] for _ in range(self.batch_size)]

        self.number_workers = option.worker_num
        self.sunset=False
        self.conv_index = 0
        self.topic_graph = self.get_topic_graph()
        self.movie1 = []
        self.num = 0
        self.num_1 = 0

        self.graph = {}
        topic_vocab = []
        with open(option.topic_redial, encoding='utf-8') as topic_file:
            for line in topic_file:
                line = line.strip('\n')
                topic_vocab.append(line)
        for topic in topic_vocab:
            self.graph[topic] = set()

    def __iter__(self):
      
        return self

    def __next__(self):
        
        for i in range(len(self.history_convs)):
            if len(self.history_convs[i]) == 0:
                if not self.sunset:
                    processed_session = self.load_processed_session()
                    if processed_session is not None and processed_session is not []:
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
        if processed_session != []:
            return processed_session
        else:
            return self.load_processed_session()

    def process(self,conversation):
        session_segs = []
        for case in conversation:
            if case['rec']==0:
                continue

            user_id = case['userid']
            context = case['contexts']
            context, context_len = clip_pad_context(context,option.context_max_len)
            context_idx = self.vocab.word2index(context)

            action = case['movie']
            action_idx = self.vocab.topic2index(action)

            topic_path = case['topic_path']

            topic_path, topic_path_len = clip_pad_sentence(topic_path,option.state_num_redial)
            topic_path_idx = self.vocab.topic2index(topic_path)

            all_topic = case['all_topic']
            all_topic, all_topic_len = clip_pad_sentence(all_topic,option.all_topic_num_redial)
            all_topic_idx = self.vocab.topic2index(all_topic)

            related_topic = self.get_related_movies(topic_path_idx, option.movie_num, action_idx)
            related_topic,related_topic_len = clip_pad_sentence(related_topic,option.movie_num)
            # related_topic = self.vocab.topic2index(related_topic)
            related_topic_idx = self.vocab.topic2index(related_topic)

            if action_idx not in related_topic_idx:
                self.num+=1

            session_segs.append([user_id,
                                 context_idx, context_len,
                                 topic_path_idx, topic_path_len,
                                 all_topic_idx, all_topic_len,
                                 action_idx,1,
                                 related_topic_idx,related_topic_len,
                                 1])

        try:
            session_segs[0][-1] = 0
        except:
            session_segs = []
        return session_segs


    def get_topic_graph(self):
        topic_graph = json.load(open('./dataset/graph_two_hop.json'))
        return topic_graph


    def get_related_movies(self, topic_path, movie_num,action ):
        related_topic = []
        for topic in topic_path:
            topic = str(topic)
            if topic in self.topic_graph:
                relation = self.topic_graph[topic]
                for r in relation:
                    related_topic.append(r)
        related_topic = list(set(related_topic))
        related_topic = related_topic[-movie_num:]
        # if action in related_topic:
        #     self.num+=1
        # else:
        #     self.num_1+=1
        return related_topic
