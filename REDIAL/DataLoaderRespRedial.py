from option import  option
from DataProcessor_Redial import clip_pad_sentence,clip_pad_context
import torch

class DataLoaderResp():
    def __init__(self,dataset,vocab):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = option.batch_size
        self.history_convs = [[] for _ in range(self.batch_size)]
        self.number_workers = option.worker_num
        self.sunset=False
        self.conv_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        for i in range(len(self.history_convs)):
            if len(self.history_convs[i]) == 0:
                if not self.sunset:
                    processed_session = self.load_processed_session()
                    if processed_session is not None and processed_session is not []:
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
            user_id = case['userid']
            context = case['contexts']
            context, context_len = clip_pad_context(context,option.context_max_len)
            context_idx = self.vocab.word2index(context)
            entities = case['entities']
            action = case['movie']
            if action!=0:
                entities.extend(action)
            entities, entities_len = clip_pad_sentence(entities, option.action_num)
            action_idx = self.vocab.topic2index(entities)
            topic_path = case['topic_path']
            topic_path, topic_path_len = clip_pad_sentence(topic_path,option.state_num_redial)
            topic_path_idx = self.vocab.topic2index(topic_path)
            resp = case['response']
            resp,resp_len = clip_pad_sentence(resp,option.r_max_len,sos=option.BOS_RESPONSE,eos=option.EOS_RESPONSE)
            resp_idx = self.vocab.word2index(resp)
            session_segs.append([user_id,
                                 context_idx, context_len,
                                 topic_path_idx, topic_path_len,
                                 action_idx,entities_len,
                                 resp_idx,resp_len,
                                 1])
        try:
            session_segs[0][-1] = 0
        except:
            session_segs = []
        return session_segs