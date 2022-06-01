from option import option
import pickle
import json as js
from tqdm import tqdm
from nltk import word_tokenize
from collections import defaultdict

class DataSet():
    def __init__(self,args):
        super(DataSet, self).__init__()
        self.args = args
        self.text_dict = pickle.load(open('dataset/text_dict.pkl', 'rb'))
        self.userSet = set()
        self.topic_vocab = []
        self.kg = pickle.load(open('./dataset/subkg.pkl','rb'))
        self.edge_list, self.n_relation = self._edge_list(self.kg, 64368, hop=2)
        with open(option.topic_redial, encoding='utf-8') as topic_file:
            for line in topic_file:
                line = line.strip('\n')
                self.topic_vocab.append(line.strip())

    def get_dialog(self,task):
        if self.args.processed:
            if task == 'rec':
                with open('./dataset/train_rec.pkl','rb+') as train_set:
                    train = pickle.load(train_set)

                with open('./dataset/valid_rec.pkl','rb+') as valid_set:
                    valid = pickle.load(valid_set)

                with open('./dataset/test_rec.pkl', 'rb+') as test_set:
                    test = pickle.load(test_set)
            elif task == 'gene':
                with open('./dataset/train_resp.pkl', 'rb+') as train_set:
                    train = pickle.load(train_set)

                with open('./dataset/valid_resp.pkl', 'rb+') as valid_set:
                    valid = pickle.load(valid_set)

                with open('./dataset/test_resp.pkl', 'rb+') as test_set:
                    test = pickle.load(test_set)
            else:
                print("task must in {} or {} !".format("rec","gene"))

            all = [train,valid, test]
            users = []
            for dataset in all:
                for data in dataset:
                    for case in data:
                        user_id = case['userid']
                        if user_id not in users:
                            user_id = int(user_id)
                            users.append(user_id)
            user_cont = max(users)+1

            train_set.close()
            valid_set.close()
            test_set.close()
            return train, valid, test, users, user_cont
        else:
            train_data = open('./dataset/train_redial.jsonl','rb+')
            valid_data = open('./dataset/valid_redial.jsonl','rb+')
            test_data = open('./dataset/test_redial.jsonl','rb+')
            id2entity = js.load(open('./dataset/mid2name_redial.json'))
            entity2entityId = pickle.load(open('./dataset/entity2entityId.pkl', 'rb'))
            self.topic_num = {}
            def _excute_data(conversations):
                convs = []
                for conversation in tqdm(conversations):
                    conversation = js.loads(conversation.strip())
                    seekerid = conversation['initiatorWorkerId']
                    recommenderid = conversation["respondentWorkerId"]
                    contexts = conversation['messages']
                    movies = conversation['movieMentions']
                    last_id = None
                    context_list = []
                    topic_path = []
                    all_topic = []
                    last_path = []
                    entity = []
                    entities = []
                    for context in contexts:
                        #
                        token_text,movie_rec,topics=_tokenize_sentece(context['text'],movies)
                        entity = list(set(topics).difference(set(movie_rec)))
                        if len(context_list) == 0:
                            context_dict = {'text': token_text,'topic_path':topic_path.copy(),
                                            'user': context['senderWorkerId'], 'movie': movie_rec,'topics':entity}
                            context_list.append(context_dict)
                            last_id = context['senderWorkerId']
                        else:
                            if context['senderWorkerId'] == last_id:
                                context_list[-1]['text'] += token_text
                                context_list[-1]['movie'] += movie_rec
                                context_list[-1]['topic_path'] = last_path
                                context_list[-1]['topics'] += entity
                            else:
                                last_path = topic_path.copy()
                                context_dict = {'text': token_text,'topic_path':last_path,
                                                'user': context['senderWorkerId'], 'movie': movie_rec,'topics':entity}
                                context_list.append(context_dict)
                                last_id = context['senderWorkerId']
                        for topic in topics:
                            topic_path.append(topic)
                    for topic in topic_path:
                        if topic not in all_topic:
                            all_topic.append(topic)
                    cases = []
                    context = []
                    for context_dict in context_list:
                        if context_dict['user'] == recommenderid and len(contexts) > 0:
                            response = context_dict['text']
                            topic_path = context_dict['topic_path']
                            entities = context_dict['topics']
                            cases.append({'userid':seekerid,'contexts': context.copy(), 'response': response,'topic_path':topic_path,
                                               'all_topic':all_topic,'entities':entities,'movie': context_dict['movie'], 'rec': 1})

                            context.append(context_dict['text'])
                        else:
                            context.append(context_dict['text'])
                    convs.append(cases)
                return convs
            def _tokenize_sentece(sentence,movies):
                topics = []
                if sentence in self.text_dict:
                    topic = self.text_dict[sentence]
                    for t in topic:
                        if t in self.topic_vocab:
                            topics.append(t)

                token_text = word_tokenize(sentence)
                num = 0
                token_text_com = []

                while num < len(token_text):
                    if token_text[num] == '@' and num + 1 < len(token_text):
                        movie = token_text[num] + token_text[num + 1]
                        token_text_com.append(movie)
                        if movie in self.topic_vocab:
                            topics.append(movie)
                        num += 2
                    else:
                        token_text_com.append(token_text[num])
                        num += 1
                movie_rec = []
                for word in token_text_com:
                    if word[1:] in movies and word in self.topic_vocab:
                        movie_rec.append(word)

                return token_text_com, movie_rec, topics

            train = _excute_data(train_data)
            valid = _excute_data(valid_data)
            test = _excute_data(test_data)

            return train, valid, test, self.userSet, len(self.userSet)

    def _edge_list(self, kg, n_entity, hop):
        edge_list = []
        for h in range(hop):
            for entity in range(n_entity):
                edge_list.append((entity, entity, 185))
                if entity not in kg:
                    continue
                for tail_and_relation in kg[entity]:
                    if entity != tail_and_relation[1] and tail_and_relation[0] != 185:  # and tail_and_relation[0] in EDGE_TYPES:
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

def clip_pad_context(context,
                     max_len,
                     bos = option.BOS_CONTEXT,
                     eos = option.EOS_CONTEXT,
                     pad = option.PAD_WORD,
                     sent = option.SENTENCE_SPLITER
                     ):
    sentence = []
    for turn in context[:-1]:
        turn = turn + [sent]
        sentence = sentence + turn
    if context:
        sentence = sentence + context[-1]

    real_len = len(sentence)

    if real_len > max_len:
        sentence = sentence[-max_len:]
    else:
        sentence = sentence + [pad] * (max_len - real_len)
    return sentence, real_len

def clip_pad_sentence(sentence,
                      max_len,
                      sos=None,
                      eos=None,
                      pad = option.PAD_WORD,
                      save_prefix=False,
                      pad_suffix=True,
                      return_length=True):

    ml = max_len
    if eos is not None:
        ml = ml - 2
    if save_prefix:
        sentence = sentence[:ml]
    else:
        sentence = sentence[-ml:]
    if eos is not None:
        sentence = [sos] + sentence
        sentence = sentence + [eos]

    length = None
    if return_length:
        length = len(sentence)
    if pad_suffix:
        sentence += [pad] * (max_len - len(sentence))
    else:
        sentence = [pad] * (max_len - len(sentence)) + sentence

    if not return_length:
        return sentence
    return sentence, length