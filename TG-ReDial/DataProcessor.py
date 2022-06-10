from option import option
import pickle
import json as js
from tqdm import tqdm
import re
import ipdb
from transformers import BertTokenizer

class DataSet():
    def __init__(self,args):
        super(DataSet, self).__init__()
        self.dataset_file = option.dataset_file
        self.args = args
        self.userSet = set()
        self.topics = self.get_topics()
        self.final_topic = js.load(open('./dataset/final_topic.json'))

    def get_dialog(self,task):
        if self.args.processed:
            if task=='rec':
                with open('./dataset/train_movie.pkl','rb+') as train_set:
                    train = pickle.load(train_set)
                with open('./dataset/valid_movie.pkl','rb+') as valid_set:
                    valid = pickle.load(valid_set)
                with open('./dataset/test_movie.pkl', 'rb+') as test_set:
                    test = pickle.load(test_set)
            elif task=='topic':
                with open('./dataset/train_topic.pkl', 'rb+') as train_set:
                    train = pickle.load(train_set)
                with open('./dataset/valid_topic.pkl', 'rb+') as valid_set:
                    valid = pickle.load(valid_set)
                with open('./dataset/test_topic.pkl', 'rb+') as test_set:
                    test = pickle.load(test_set)
            elif task=='gene':
                with open('./dataset/train_resp.pkl', 'rb+') as train_set:
                    train = pickle.load(train_set)
                with open('./dataset/valid_resp.pkl', 'rb+') as valid_set:
                    valid = pickle.load(valid_set)
                with open('./dataset/test_resp.pkl', 'rb+') as test_set:
                    test = pickle.load(test_set)
            else:
                print("task must in topic, rec and gene")
            all = [train, test]
            users = []
            for dataset in all:
                for data in dataset:
                    user_id = data[0]
                    if user_id not in users:
                        user_id = int(user_id)
                        users.append(user_id)
            user_cont = max(users)+1
            return train, valid, test, users, user_cont

        else:
            train_data = pickle.load(open('./dataset/train_data.pkl','rb+'))[:]
            valid_data = pickle.load(open('./dataset/valid_data.pkl', 'rb+'))[:]
            test_data = pickle.load(open('./dataset/test_data.pkl', 'rb+'))[:]
            def _excute_data(conversations):
                convs = []
                for conversation in tqdm(conversations):
                    user_id,conv_id,utterances,topic_thread,movies = conversation['user_id'],conversation['conv_id'],\
                                                              conversation['messages'],conversation['goal_path'],conversation['mentionMovies']
                    conv = []
                    self.userSet.add(user_id)
                    conv.append(user_id)
                    contents_word = []
                    contents_token = []
                    word2tokens = []
                    length = []
                    states = []
                    alltopic = []
                    ks = 1
                    for utterance in utterances:
                        processed_sentence = []
                        utter_round,role,content = int(utterance['local_id']),utterance['role'],utterance['content']
                        goal = topic_thread[utter_round] if utter_round!=1 else [0]
                        action, topics = self.get_action(goal, movies, utter_round)
                        if utter_round != 1:
                            final_topic = self.get_final_topic(conv_id, utter_round)
                            final_states = states.copy()
                            for topic in final_topic:
                                final_states.append(topic)
                            word_level, word2token, leng, ks  = self.tokenize_sentence(content,movies,utter_round,ks)
                        else:
                            action = []
                            final_states = states
                            word_level, word2token, leng, ks = self.tokenize_sentence(content,movies,utter_round,ks)
                        word2tokens.extend(word2token)
                        length.extend(leng)
                        length.append(1)
                        contents_word.append(word_level)
                        processed_sentence.append(final_states.copy())
                        processed_sentence.append(action)
                        processed_sentence.append([utter_round])
                        conv.append(processed_sentence)
                        for topic in topics:
                            states.append(topic)
                    for topic in states:
                        if topic not in alltopic:
                            alltopic.append(topic)
                    conv.append(contents_word)
                    conv.append(contents_token)
                    conv.append(word2tokens)
                    conv.append(length[:-1])
                    conv.append(alltopic)
                    convs.append(conv)
                return convs
            train = _excute_data(train_data)
            valid = _excute_data(valid_data)
            test = _excute_data(test_data)
            return train, valid, test, self.userSet, len(self.userSet)

    def tokenize_sentence(self,sentence: str,movies,turn,ks):
        if turn in movies:
            assert "《" in sentence and "》" in sentence
            movie_id = movies[turn][0]
            con = re.sub(r'《(.*)》', '<movie>', sentence)
            split_content = con.split('<movie>')
            sentence = split_content[0] + '<movie>' + split_content[1]
        processed_sentence = []
        while (sentence):
            flag = 0
            for topic in self.topics:
                if topic in sentence:
                    idx = sentence.index(topic)
                    if idx == 0:
                        flag = 1
                        processed_sentence.append(topic)
                        sentence = sentence[len(topic):]
                        continue
            if turn in movies and movies[turn][0] in sentence:
                if sentence.index(movies[turn][0]) == 0:
                    flag = 1
                    processed_sentence.append(movies[turn][0])
                    sentence = sentence[len(movies[turn][0]):]
            if flag == 0:
                word = sentence[0]
                processed_sentence.append(word)
                sentence = sentence[1:]
        word2token = []
        for word in processed_sentence:
            if word == '<movie>':
                length = 3
            else:
                length = len(word)
            word2token.append([ks+j for j in range(length)])
            ks+=length
        leng = []
        for word in word2token:
            leng.append(len(word))
        word2token_pad = []
        for i in range(len(word2token)):
            word = word2token[i]
            length = leng[i]
            pad_token = word + [0]*(10-length)
            word2token_pad.append(pad_token)
        return processed_sentence, word2token_pad, leng, ks

    def get_action(self,goals,movies,utter_round):
        action = []
        topic_path = []
        goal = goals[1:]
        if '反馈' in goal:
            assert goal[0] == '反馈'
            goal = goal[2:4]
        if '谈论' in goal and '请求推荐' in goal:
            goal = goal[2:4]
        if len(goal) == 2:
            action_type = goal[0]
            topics = goal[1]
            if '推荐电影' in action_type:
                if isinstance(topics, str):
                    action.append(action_type)
                    movie = movies[utter_round][0]
                    action.append('<movie>')
                    if '拒绝' not in action_type:
                        topic_path.append(movie)
                elif isinstance(topics, list):
                    for topic in topics:
                        action.append(action_type)
                        action.append('<movie>')
                        if '拒绝' not in action_type:
                            topic_path.append(topic)
            else:
                if isinstance(topics, str):
                    action.append(action_type)
                    action.append(topics)
                    topic_path.append(topics)
                elif isinstance(topics, list):
                    for topic in topics:
                        action.append(action_type)
                        action.append(topic)
                        topic_path.append(topic)
        elif len(goal) == 4:
            for i in range(0, 4, 2):
                action_type = goal[i]
                topics = goal[i + 1]

                if '推荐电影' in action_type:
                    if isinstance(topics, str):
                        action.append(action_type)
                        movie = movies[utter_round][0]
                        action.append('<movie>')
                        if '拒绝' not in action_type:
                            topic_path.append(movie)
                    if isinstance(topics, list):
                        for topic in topics:
                            action.append(action_type)
                            action.append('<movie>')
                            if '拒绝' not in action_type:
                                topic_path.append(topic)
                else:
                    if isinstance(topics, str):
                        action.append(action_type)
                        action.append(topics)
                        movie = movies[utter_round][0]
                        action.append(movie)
                        if '拒绝' not in action_type:
                            topic_path.append(topics)
                    if isinstance(topics, list):
                        for topic in topics:
                            action.append(action_type)
                            action.append(topic)
                            if '拒绝' not in action_type:
                                topic_path.append(topic)
        return action,topic_path

    def get_state(self,action):
        state = []
        delete_state = []

        action_len = len(action)
        for k in range(0,action_len,2):
            action_type = action[k]
            topic = action[k+1]
            if '拒绝' in action_type:
                delete_state.append(topic)
            else:
                state.append(topic)
        return state,delete_state

    def get_final_topic(self, conv_id, utter_id):
        kw_list = []
        conv_id = str(conv_id)
        utter_id = str(utter_id)
        identity =  conv_id + '/' + utter_id
        if identity in self.final_topic:
            kw_list = self.final_topic[identity]
        return kw_list

    def get_topics(self):
        topic_file = open(option.topic_file, encoding='utf-8')
        topic_vocab = []
        for line in topic_file.readlines():
            line = line.strip('\n')
            topic_vocab.append(line)
        return topic_vocab

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

def clip_pad_context(context,max_len,pad = option.PAD_WORD,sent = option.SENTENCE_SPLITER):
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