import json
import ipdb
from option import  option
from enum import Enum
from DataProcessor import clip_pad_sentence,clip_pad_context
from Vocab import Vocab
import torch
import pickle
import json as js

class DataLoaderTopic():
    def __init__(self,dataset,vocab):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = option.batch_size
        self.history_convs = [ [] for _ in range(self.batch_size)]
        self.number_workers = option.worker_num
        self.sunset=False
        self.conv_index = 0
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
        all_topic, all_topic_len = clip_pad_sentence(all_topics, option.all_topic_num)
        all_topic = self.vocab.topic2index(all_topic)
        utterances = conversation[1:-2]
        uttr_len = len(utterances)
        pv_action = []
        for i in range(2,uttr_len,2):
            response = utterances[i]
            action_R = response[2]
            if action_R == []:
                continue
            resp = response[0]
            resp, resp_len = clip_pad_sentence(resp,option.r_max_len,sos=option.BOS_RESPONSE,eos=option.EOS_RESPONSE)
            context = contexts[:i]
            context, context_len = clip_pad_context(context,option.context_max_len)
            state_U = response[1]
            state_U, state_U_len = clip_pad_sentence(state_U, option.state_num)
            Seeker = utterances[i - 1]
            action_U = Seeker[2]
            if action_U != []:
                pv_action = action_U
            related_topics = self.get_related_topics(pv_action,option.relation_num,action_R)
            related_topics, related_topics_len = clip_pad_sentence(related_topics,option.relation_num)
            context_idx = self.vocab.word2index(context)
            state_U = self.vocab.topic2index(state_U)
            related_topics = self.vocab.topic2index(related_topics)
            a_R, a_R_len = clip_pad_sentence(action_R,option.action_num)
            a_R = self.vocab.topic2index(a_R)
            session_segs.append([id,context_idx,context_len,state_U,state_U_len,related_topics,
                                 related_topics_len,a_R,a_R_len,all_topic,all_topic_len,1])
        session_segs[0][-1] = 0
        return session_segs

    def get_topic_graph(self):
        with open('./dataset/topic_graph.json') as f:
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
            related_topics.extend(related_topic)
        return related_topics

def one_hot_scatter(indice, num_classes, dtype=torch.float):
    indice_shape = list(indice.shape)
    placeholder = torch.zeros(*(indice_shape + [num_classes]), device=indice.device, dtype=dtype)
    v = 1 if dtype == torch.long else 1.0
    placeholder.scatter_(-1, indice.unsqueeze(-1), v)
    return placeholder

