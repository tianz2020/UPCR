#coding:utf-8
import csv
import gc
import pickle
import json
import re
import ipdb
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import torch
from option import option
import sys
from math import log2,exp
# from transformers import BertTokenizer,BertModel,BertConfig


# 读 pkl文件
# relations = pickle.load(open(r'C:\Users\Administrator\Desktop\res1\idTag_1.pkl','rb+'))
# for re in relations:
#     print(re)
#     print(relations[re])

# 存 pkl文件
# with open(r'C:\Users\86176\Desktop\mywork\testwqeqw.pkl', 'wb+') as f:
#     pickle.dump(a,f)

# jieba.load_userdict(option.special_words_file)
#
# train_data = pickle.load(open('./dataset/train_data.pkl','rb+'))[:]
# valid_data = pickle.load(open('./dataset/valid_data.pkl', 'rb+'))[:]
# test_data = pickle.load(open('./dataset/test_data.pkl', 'rb+'))[:]
# data = test_data + valid_data + train_data
# for conv in data:
#     message = conv['messages']
#     for me in message:
#         content = me['content']
#         if '那天看到一个笑话笑了好久' in content:
#             print(conv)
#             break

# #
# data = train_data + valid_data + test_data
# #


# # print(data[0])
# # contents = set()
# # for conv in tqdm(data):
# #     messages = conv['messages']
# #     for message in messages:
# #         content = message['content']
# #         if '《' in content and '》' in content:
# #             # 处理电影
# #             con = re.sub(r'《(.*)》', '<movie>', content)
# #             split_content = con.split('<movie>')
# #             content = list(split_content[0])+['<movie>']+list(split_content[1])
# #         for c in content:
# #             contents.add(c)
# # topics = json.load(open('./dataset/topic2id.json'))
# # for topic in topics:
# #     contents.add(topic)
# #
# # f = open('./dataset/vocab_1.txt','w+',encoding='utf-8')
# # for content in contents:
# #     f.write(content+'\n')

'''
统计词表
'''
# topics = json.load(open('./dataset/topic2id.json'))
#
# f = open('dataset/topics.txt', 'w+',encoding='utf-8')
# # with open('./dataset/topics.txt','wb+') as f:
# for topic in topics:
#     f.write(topic+'\n')
# goals = set()


# mention_movies = set()
# for conv in data:
#     movies = conv['mentionMovies']
#     for key in movies:
#         movie = movies[key][0]
#         mention_movies.add(movie)
#
# movie_relations = []
# with open('./dataset/movie_relations.csv') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if row[0] in mention_movies:
#             movie_relations.append([row[0],row[1],row[2]])
#
# with open('./dataset/movie_with_mentions.csv','w+',newline='') as f1:
#     writer = csv.writer(f1)
#     for relation in movie_relations:
#         writer.writerow(relation)

'''
构建词表代码
'''
# def tokenize_sentence( sentence: str, goal):
#     goal_len = len(goal)
#     topics = set()
#     if '《' in sentence and '》' in sentence:
#         # 处理电影
#         con = re.sub(r'《(.*)》', '<movie>', sentence)
#         split_content = con.split('<movie>')
#         sentence = list(split_content[0])+['<movie>']+list(split_content[1])
#
#     topics.add('<movie>')
#     for i in range(0, goal_len, 2):
#         action_type = goal[i]
#         if action_type == '推荐电影':
#             continue
#         # 拿到所有的topic
#         topic = goal[i + 1]
#         if isinstance(topic, str):
#             topics.add(topic)
#         elif isinstance(topic,list):
#             for t in topic:
#                 topics.add(t)
#         else:
#             continue
#
#     # topic_idx = []
#
#     # for topic in topics:
#     #     # topic所在的index
#     #     idx = sentence.index(topic)
#     #     topic_idx.append(idx)
#     # topic_idx.sort()
#     processed_sentence = []
#     while(sentence):
#         for topic in topics:
#             if topic in sentence:
#                 idx = sentence.index(topic)
#                 if idx==0:
#                     processed_sentence.append(topic)
#                     sentence = sentence[len(topic):] if topic!='<movie>' else sentence[1:]
#                     continue
#
#         if sentence:
#             word = sentence[0]
#             processed_sentence.append(word)
#             sentence = sentence[1:]
#
#     return  processed_sentence
# words = set()
# for conv in tqdm(data):
#
#     messages = conv['messages']
#     goal_path = conv['goal_path']
#
#     for message in messages:
#         id,content = message['local_id'],message['content']
#         id = int(id)
#         if id ==1 :
#             continue
#         goal = goal_path[id]
#         goal = goal[1:]
#
#         tokenized_content = tokenize_sentence(content,goal)
#         for token in tokenized_content:
#             words.add(token)
# f = open('./dataset/vocab_1.txt','w+',encoding='utf-8')
# for word in words:
#     f.write(word+'\n')


# relations = []
#
# 拿到所有的topic
# topic_vocab = []
# topic_file = open(option.topic_file, encoding='utf-8')
# for line in topic_file.readlines():
#     line = line.strip('\n')
#     topic_vocab.append(line)
#
# origin = json.load(open('./dataset/topics2movieid.json'))
# new = json.load(open('./dataset/allgraph_allmovie.json'))
# new_1 = {}
# for topic in topic_vocab:
#     relations = new[topic]
#     new_1[topic] = relations
#
# json.dump(new_1,open('./dataset/topic2movie.json','w+'))

'''
relations_1  从topic_relation里面单独把有关电影的拿出来
'''
# name2id = {}
# id2topics = {}
# movie_id = pd.read_csv('./dataset/movie_relations.csv',usecols=[0,1,2],encoding='gbk')
# movies = movie_id.values.tolist()
# for movie in movies:
#     movie[1] = re.sub('\(\d*\)','',movie[1])
#     movie[1] = re.sub('\(上\)','',movie[1])
#     movie[1] = re.sub('\(下\)','',movie[1])
#     name2id[movie[1]] = str(movie[0])
#     id2topics[str(movie[0])] = []
#
#
# for movie in tqdm(movies):
#     id = str(movie[0])
#     tag = movie[2]
#     tag = re.sub('{', '', tag)
#     tag = re.sub('}', '', tag)
#     tag = re.sub("'", '', tag)
#     tags = tag.split(', ')
#     for t in tags:
#         if t in topic_vocab and t not in id2topics[id]:
#             id2topics[id].append(t)



# topic_relaions = pd.read_csv('./graphormer/data/topic_relations.csv')
# corpus = topic_relaions.values.tolist()
# for i in corpus:
#     topic1 = i[0]
#     topic2 = i[2]
#     if topic1 in name2id and topic2 in topic_vocab:
#         id2topics[name2id[topic1]].append(topic2)
#     if topic2 in name2id and topic1 in topic_vocab:
#         id2topics[name2id[topic2]].append(topic1)


# with open('./dataset/movie_with_mentions.csv', 'r', encoding='gbk') as movie_file:
#     movie_reader = csv.reader(movie_file)
#     for row in movie_reader:
#         id = int(row[1])
#         # 去掉原本name中的括号
#         tag = row[3]
#         tag = re.sub('{', '', tag)
#         tag = re.sub('}', '', tag)
#         tag = re.sub("'", '', tag)
#         tags = tag.split(', ')
#         for t in tags:
#             if t in topic_vocab and t not in id2topics[id]:  # 只把在topics里面的tag加上
#                 id2topics[id].append(t)
# topics2id = {}
# for topic in topic_vocab:
#     topics2id[topic] = []
#
# for movie,topics in id2topics.items():
#     for topic in topics:
#         topics2id[topic].append(movie)
#
#
# json.dump(id2topics,open('./dataset/movieid2topics.json','w+'))
# json.dump(topics2id,open('./dataset/topics2movieid.json','w+'))

#     if topic1!=topic2 and [topic1,topic2] not in relations and [topic2,topic1] not in relations:
#         relations.append([topic1,topic2])
#
# with open('./dataset/topic_relations_1.csv','w+',encoding='utf-8',newline='') as topicfile:
#     writer = csv.writer(topicfile)
#     for relation in relations:
#         writer.writerow(relation)



# name2id = {}
# movie_id = pd.read_csv('./dataset/movie_with_mentions.csv',usecols=[1,2],encoding='gbk')
# movies = movie_id.values.tolist()
# for movie in movies:
#     movie[1] = re.sub('\(\d*\)','',movie[1])
#     movie[1] = re.sub('\(上\)', '', movie[1])
#     movie[1] = re.sub('\(下\)', '', movie[1])
#     name2id[movie[1]] = movie[0]


'''
topic_graph   
'''
# topic_only = {}
#
# with open('./dataset/topic_relations.csv',encoding='utf-8') as f:
#     reader = csv.reader(f)
#     for line in reader:
#         t1 = line[0]
#         t2 = line[2]
#         if t1 not in topic_only:
#             topic_only[t1] = [t2]
#         else:
#             if t2 not in topic_only[t1]:
#                 topic_only[t1].append(t2)
#         if t2 not in topic_only:
#             topic_only[t2] = [t1]
#         else:
#             if t1 not in topic_only[t2]:
#                 topic_only[t2].append(t1)

# for conv in data:
#     utterances = conv[1:-1]
#     utter_len = len(utterances)
#     for i in range(1,utter_len-1,2):
#         seeker = utterances[i][2]
#         rec = utterances[i+1][2]
#         if '邱淑贞' in seeker:
#             print(seeker)
#             print(rec)
#         if seeker == [] or rec == []:
#             continue
#         seeker_len = len(seeker)
#         for k in range(1,seeker_len,2):
#             type = seeker[k-1]
#             if '拒绝' in type:
#                 continue
#             topic = seeker[k]
#             if topic not in topic_only:
#                 topic_only[topic] = [topic]
#             else:
#                 exist = topic_only[topic]
#                 for target in rec:
#                     if target in ['谈论','请求推荐']:
#                         continue
#                     else:
#                         if target not in exist:
#                             exist = [target] + exist
#                             topic_only[topic] = exist
#                         else:
#                             exist.remove(target)
#                             exist = [target] + exist
#                             topic_only[topic] = exist
#
# topic_graph = json.load(open('./dataset/topic_graph.json'))
# print(topic_graph['邱淑贞'])
# i = 0
# topic2movie = json.load(open('./dataset/topic_graph.json'))
# for k in topic2movie:
#     if len(topic2movie[k]) > 100:
#         i += 1
# print(i)



# vocab = Vocab()
# a,b,topic_vocab,d = vocab.get_vocab()
#
# topic2topic = json.load(open('./dataset/topic_graph.json'))
# movie2topic = json.load(open('./dataset/movieid2topics.json'))
# relations = []
# topic2movie = json.load(open('./dataset/topics2movieid.json'))
# for topic in topic_vocab:
#     if topic in topic2topic:
#         related_topics = topic2topic[topic]
#         relations.append(related_topics)
#     elif topic in movie2topic:
#         related_topics = movie2topic[topic]
#         relations.append(related_topics)
#     else:
#         pass


# f = pickle.load(open('./dataset/train_movie.pkl','rb+'))
# print(f[0])
# topic_graph = json.load(open('./dataset/topics2movieid.json'))

#
# train = pickle.load(open('./dataset/train_movie.pkl','rb+'))
# valid = pickle.load(open('./dataset/valid_movie.pkl','rb+'))
# test = pickle.load(open('./dataset/test_movie.pkl', 'rb+'))
# all = train + valid + test
# all_len = len(all)
# j = 0
# q = 0
# graph = {}
# for conv in tqdm(all,total=all_len):
#     utterances = conv[1:-1]
#     uttr_len = len(utterances)
#     for i in range(2, uttr_len, 2):
#         response = utterances[i]
#         action_R = response[2]
#         if action_R == []:
#             continue
#         movie = action_R[1]
#         if movie not in graph:
#             graph[movie] = [movie]
#
#         Seeker = utterances[i - 1]
#         topic_path = Seeker[1]
#         new_path = []
#         for t in topic_path:
#             if t not in new_path :
#                 new_path.append(t)
#         new_path = new_path[-int(i / 2.5):]
#
#         for topic in new_path:
#             if topic not in topic_vocab:
#                 if topic in name2id:
#                     topic = name2id[topic]
#
#             if topic not in graph:
#                 graph[topic] = []
#
#             if movie not in graph[topic]:
#                 graph[topic] = [movie] + graph[topic]

# json.dump(graph,open('./dataset/topics2movieid_baseed_on_data.json','w+'))

# j = 0
# # origin = json.load(open('./dataset/topics2movieid.json'))
# vd = json.load(open('./dataset/full_graph.json'))
#
#
# for m in tqdm(id2topics):
#     topics = id2topics[m]
#     for topic in topics:
#         if m not in vd[topic]:
#             vd[topic].append(m)
#
# json.dump(vd,open('./dataset/allgraph_allmovie.json','w+'))

# topic_graph = json.load(open('./dataset/topic_graph_1123.json'))
# def get_related_topics( action_U, relation_num):
#     related_topics = []
#     a_len = len(action_U)
#     for k in range(0, a_len, 2):
#         action_type = action_U[k]
#         topic = action_U[k + 1]
#         if '拒绝' in action_type:
#             assert a_len > 1
#             continue
#         related_topic = topic_graph[topic][0:int(2 * relation_num / a_len)]
#         related_topics.extend(related_topic)
#     return related_topics
#
#
# train = pickle.load(open('./dataset/train_topic_1123.pkl','rb+'))
# valid = pickle.load(open('./dataset/valid_topic_1123.pkl','rb+'))
# test = pickle.load(open('./dataset/test_topic_1123.pkl', 'rb+'))
# all = train + valid + test
# all_len = len(all)
#
# num = 0
# for conv in tqdm(all):
#     utterances = conv[1:-2]
#     uttr_len = len(utterances)
#     pv_action = []
#     for i in range(2, uttr_len, 2):
#         Seeker = utterances[i - 1]
#         action_U = Seeker[2]
#         response = utterances[i]
#         action_R = response[2]
#         if action_U != []:
#             pv_action = action_U
#         related_topics = get_related_topics(pv_action,150)
#
#         if action_U == []:
#             action_R_len = len(action_R)
#             for j in range(0,action_R_len,2):
#                 ar = action_R[j+1]
#                 if ar not in related_topics:
#                     action_U_len = len(pv_action)
#                     for d in range(0,action_U_len,2):
#                         au = pv_action[d+1]
#                         if '拒绝' in pv_action[d]:
#                             continue
#
#                         topic_graph[au] = [ar] + topic_graph[au]
#
# json.dump(topic_graph,open('./dataset/topic_graph_1124.json','w+'))

# topic_graph = {}
# for topic in topic_vocab:
#     topic_graph[topic] = []
#
# for conv in tqdm(data):
#     utterances = conv[1:-2]
#     uttr_len = len(utterances)
#
#     for i in range(2, uttr_len, 2):
#         response = utterances[i]  # R_t
#         action_R = response[2]
#         if action_R == []:
#             break
#
#         Seeker = utterances[i - 1]
#         action_U = Seeker[2]
#         a_len = len(action_U)
#         for k in range(0,a_len,2):
#             action_type = action_U[k]
#             topic_u = action_U[k+1]
#             if '拒绝' in action_type:
#                 assert a_len > 1
#                 continue
#
#             ar_len = len(action_R)
#
#             for j in range(0,ar_len,2):
#                 topic_r = action_R[j+1]
#                 if topic_r not in topic_graph[topic_u]:
#                     topic_graph[topic_u].append(topic_r)
#
# json.dump(topic_graph,open('./dataset/topic_graph_1123.json','w+'))


# pv_m = torch.randn([2,2,4])
# final = torch.tensor([0,1])
# final = final.unsqueeze(1).unsqueeze(1)
# final = final.expand(-1,2,-1).expand(-1,-1,4)
# pv_m = pv_m.mul(final)
# pv_m[:,:,2] = 1 - torch.sum(pv_m,-1)
# print(pv_m)


# test_topic = pickle.load(open('./dataset/test_topic_1127.pkl','rb+'))
# test_resp = pickle.load(open('./dataset/test_resp.pkl','rb+'))
# print(test_topic[0])
# print(test_resp[0])

# test = json.load(open('./dataset/test_resp.json'))
# for i,case in test.items():
#     ipdb.set_trace()
#     print(case)

a = torch.tensor([  [ 0.5 , 0.3 , 0.2   ] ,
                    [0.1, 0.7,0.2 ]] )
print(a.shape)
b = torch.tensor([   [ 2,1  ]   ,[0,1]   ]    )

for i in range(2):
    a[i][b[i]] /= 5
print(a)