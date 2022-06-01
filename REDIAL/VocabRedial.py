import json as js
import torch
from option import option
import numpy as np

class Vocab(object):
    def __init__(self,word_vocab=False,topic_vocab=False):
        super(Vocab, self).__init__()
        self.word_list,self.word_len,self.topic_list,self.topic_len= self.get_vocab()
        self.word2idx = dict(zip(self.word_list, range(len(self.word_list))))
        self.idx2word = { id:word for word,id in self.word2idx.items() }
        self.topic2idx = dict(zip(self.topic_list, range(len(self.topic_list))))
        self.idx2topic = {id:word for word,id in self.topic2idx.items()}
        self.word_vocab = word_vocab
        self.topic_vocab = topic_vocab

    def get_vocab(self):
        RESERVED_WORDS = [ option.PAD_WORD,option.BOS_PRE, option.BOS_PRO,
                           option.UNK_WORD,option.BOS_ACTION]
        topic_vocab = []
        with open(option.topic_redial, encoding='utf-8') as topic_file:
            for line in topic_file:
                line = line.strip('\n')
                topic_vocab.append(line)
        topic_vocab = topic_vocab + RESERVED_WORDS
        topic_len = len(topic_vocab)
        RESERVED_WORDS_1 = [option.PAD_WORD,option.UNK_WORD,option.SENTENCE_SPLITER,
                            option.BOS_RESPONSE,option.EOS_RESPONSE]
        word_vocab = []
        with open(option.vocab_redial, encoding='utf-8') as vocab_file:
            for line in vocab_file.readlines():
                line = line.strip('\n')
                word_vocab.append(line)
        word_vocab = word_vocab + RESERVED_WORDS_1
        word_len = len(word_vocab)
        return word_vocab, word_len, topic_vocab, topic_len

    def word2index(self,word):
        unk_id = self.word2idx.get('[UNK]')
        if isinstance(word, str):
            return self.word2idx.get(word, unk_id)
        elif isinstance(word, list):
            return [self.word2index(w) for w in word]
        else:
            raise ValueError("wrong type {}".format(type(word)))

    def index2word(self,index):
        if isinstance(index, int):
            if index < len(self.word_list):
                return self.word_list[index]
            else:
                raise ValueError("{} is out of {}".format(index, len(self.word_list)))
        elif isinstance(index, np.ndarray):
            index = index.tolist()
            return [self.index2word(i) for i in index]
        elif isinstance(index, list):
            return [self.index2word(i) for i in index]
        else:
            raise ValueError("wrong type {}".format(type(index)))

    def topic2index(self,topic):
        unk_id = self.topic2idx.get('[UNK]')
        if isinstance(topic, str):
            return self.topic2idx.get(topic, unk_id)
        elif isinstance(topic, list):
            return [self.topic2index(w) for w in topic]
        elif isinstance(topic,int):
            return topic
        elif topic is None:
            return self.topic2idx.get(option.PAD_WORD)
        else:
            raise ValueError("wrong type {}".format(type(topic)))

    def index2topic(self,index):
        if isinstance(index, int):
            if index < len(self.topic_list):
                return self.topic_list[index]
            elif index == len(self.topic_list):
                return None
            else:
                raise ValueError("{} is out of {}".format(index, len(self.word_list)))
        elif isinstance(index, np.ndarray):
            index = index.tolist()
            return [self.index2topic(i) for i in index]
        elif isinstance(index, list):
            return [self.index2topic(i) for i in index]
        else:
            raise ValueError("wrong type {}".format(type(index)))

    def item_in(self, word):
        if self.word_vocab:
            return self.word2index(word)
        elif self.topic_vocab:
            return self.topic2index(word)
        else:
            raise ValueError("word_vocab or topic_vocab must be true")

    def __len__(self,word=False,topic=False):
        if word:
            return self.word_len
        elif topic:
            return self.topic_len
        else:
            raise ValueError("word_vocab or topic_vocab must be true")

    def vocab_transfer(self):
        glo2loc = []
        for word in self.word_list:
            glo2loc.append(self.topic2index(word))

        loc2glo = []
        for index,topic in enumerate(self.topic_list):
            loc2glo.append(self.word2index(topic))
        return glo2loc, loc2glo

    def get_word_pad(self):
        return self.word2index('[PAD]')

    def get_topic_pad(self):
        return self.topic2index('[PAD]')

    def topic_num(self):
        return self.topic_len

    def movie_num(self):
        non_movie = self.topic2index('<movie>') + 1
        movienum = self.topic_num() - non_movie
        return movienum

    def get_topic_relations(self):
        topic2topic = js.load(open('./dataset/topic_graph.json'))
        relations = torch.zeros([self.topic_len,self.topic_len],dtype=torch.float)
        for i, topic in enumerate(self.topic_list):
            related_topics = []
            if topic in topic2topic:
                related_topics = topic2topic[topic]
            related_topics = self.topic2index(related_topics)
            relations[i, related_topics] = 1.0
        relations = relations.cuda()
        relations[:, 0] = 0.0
        relations = relations.unsqueeze(0).expand(option.batch_size, -1, -1)
        return relations