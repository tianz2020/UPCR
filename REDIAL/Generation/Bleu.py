#coding=utf-8

import argparse
import json
import sys
from nltk.translate import bleu_score as nltkbleu
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import ipdb
import re

from option import option

mid2name = json.load(open('./dataset/mid2name_redial.json'))
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
#    def remove_articles(text):
#        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    #return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(lower(s)))

def bleu(tokenized_gen, tokenized_tar):
    print_num = 0
    bleu, count = 0, 0
    bleu_sum = 0
    for sen, tar in zip(tokenized_gen, tokenized_tar):

        for j,word in enumerate(sen):
            if word == option.EOS_RESPONSE:
                sen = sen[:j]
                break
        tar = tar[1:]  # 删掉BOS
        for k,word in enumerate(tar):
            if word == option.EOS_RESPONSE:
                tar = tar[:k]
                break

        full_sen_gen = ''
        full_sen_gth = ''

        for word in sen:
            if '@' in word:
                movie_id = word[1:]
                if movie_id in mid2name:
                    movie = mid2name[movie_id]
                    full_sen_gen += movie
                else:
                    full_sen_gen += word
            else:
                full_sen_gen += word
            full_sen_gen += " "
        for word in tar:
            if '@' in word:
                movie_id = word[1:]
                if movie_id in mid2name:
                    movie = mid2name[movie_id]
                    full_sen_gth += movie
                else:
                    full_sen_gth += word
            else:
                full_sen_gth += word
            full_sen_gth += " "

        if print_num<5:
            print("ground_truth:{}".format(full_sen_gth))
            print("generation:{}".format(full_sen_gen))
            sys.stdout.flush()
            print_num += 1

        bleu = nltkbleu.sentence_bleu(
            [normalize_answer(full_sen_gth).split(" ")],
            normalize_answer(full_sen_gen).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method7,
        )
        bleu_sum += bleu
        count += 1

    return bleu_sum/count

def main():
    # gth = [['[s_response>]', '还', '有', '一', '部', '没', '有', '在', '大陆', '公', '映', '的', '警察', '题', '材', '电', '影', '，', '我', '非', '常', '推', '荐', '<movie>', '，', '那', '个', '警', '官', '说', '得', '好', '，', '不', '要', '依', '靠', '饥饿', '的', '人', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '高', '智', '商', '人类', '是', '发', '展', '的', '核', '心', '竞争', '力', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '你', '可', '以', '看', '<movie>', '，', '胜', '在', '原', '声', '、', '巴黎', '的', '街', '景', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '看', '看', '<movie>', '吧', '，', '我', '觉', '得', '四', '部', '里', '边', '觉', '得', '最', '搞笑', '的', '一', '部', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '推', '荐', '你', '看', '<movie>', '，', '灰', '姑', '娘', '与', '丑', '小', '鸭', '合', '体', '，', '都市', '童话', '，', '还', '不', '错', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '那', '你', '可', '以', '看', '看', '<movie>', '，', '细', '腻', '的', '画', '质', '，', '流', '畅', '的', '过', '度', '，', '恰', '当', '的', '讽刺', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '<movie>', '满', '足', '你', '，', '高', '楼', '多', '角', '度', '摄影', '展', '示', '惊险', '，', '双', '线', '叙', '事', '推', '迟', '悬念', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '有', '，', '吐', '血', '推', '荐', '<movie>', '，', '淡', '淡', '的', '温暖', '的', '关', '于', '亲情', '的', '电', '影', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '又', '对', '历史', '这', '门', '学', '科', '心动', '了', '呀', '，', '可', '以', '说', '你', '是', '见', '异', '思', '迁', '吗', '？', '哈', '哈', '哈', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '哈', '哈', '，', '十', '分', '推', '荐', '你', '看', '<movie>', '，', '这', '是', '张艺谋', '的', '片', '子', '中', '我', '最', '喜', '欢', '的', '一', '部', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '强', '力', '推', '荐', '<movie>', '，', '韦', '小', '宝', '和', '周星驰', '这', '俩', '人', '就', '很', '搭', '啊', '！', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '你', '会', '喜', '欢', '<movie>', '的', '，', '家庭', '剧', '，', '主', '要', '讲', '战', '后', '的', '日', '本', '，', '用', '时', '间', '流', '转', '，', '跳', '轴', '的', '手', '法', '去', '讲', '述', '的', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '每', '个', '人', '都', '有', '内', '心', '最', '脆', '弱', '的', '一', '面', '，', '我', '这', '里', '倒', '是', '有', '一', '些', '温情', '的', '电', '影', '给', '你', '看', '看', '吧', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '极', '力', '推', '荐', '<movie>', '给', '你', '，', '感', '觉', '有', '时', '善', '恶', '的', '两', '极', '就', '是', '人性', '的', '上', '下', '限', '，', '上', '限', '是', '恶', '反', '之', '下', '限', '是', '善', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '哈', '哈', '，', '其', '实', '动画', '中', '也', '有', '很', '多', '励志', '角', '色', '的', '呢', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[s_response>]', '<movie>', '非', '常', '适', '合', '你', '，', '感', '觉', '那', '时', '的', '导演', '拍', '得', '现实', '主', '义', '大', '都', '很', '棒', ' ', '。', '[/s_response]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]
    # gen = [['特', '别', '推', '荐', '你', '看', '<movie>', '，', '有', '关', '于', '大陆', '警察', '的', '电', '影', '，', '有', '兴趣', '吗', '，', '可', '以', '去', '看', '看', '，', '题', '材', '很', '不', '错', '。', '[/s_response]', '版', '的', '阿', '婆', '总', '是', '那', '么', '牛', '逼', '。', '[/s_response]', '。', '[/s_response]', '。', '[/s_response]'], ['这', '样', '啊', '，', '你', '可', '真', '是', '个', '厉', '害', '的', '人', '。', '其', '实', '这', '个', '话', '题', '也', '挺', '有', '意', '思', '的', '。', '[/s_response]', '导', '少', '个', '哪', '些', '哪', '种', '口', '味', '分', '喜', '欢', '？', '[/s_response]', '吗', '？', '[/s_response]', '模', '式', '。', '[/s_response]', '[/s_response]'], ['你', '会', '喜', '欢', '<movie>', '的', '，', '推', '荐', '你', '看', '，', '这', '电', '影', '是', '我', '最', '爱', '的', '法国', '战争', '电', '影', '之', '一', '。', '[/s_response]', '的', '原', '来', '就', '为', '女', '主', '吧', '。', '[/s_response]', '.', '[/s_response]', '也', '帅', '到', '了', '。', '[/s_response]', '。', '[/s_response]', '。', '[/s_response]'], ['你', '会', '喜', '欢', '<movie>', '的', '，', '星', '爷', '的', '幽默', '，', '黑', '色', '喜剧', '。', '[/s_response]', '，', '祖', '贤', '好', '短', '。', '[/s_response]', '的', '喜剧', '。', '[/s_response]', '正', '在', '轻松', '吧', '？', '[/s_response]', '。', '[/s_response]', '个', '之', '这', '部', '影片', '真', '的', '很', '搞笑', '。', '[/s_response]', '。', '[/s_response]', '。'], ['你', '会', '喜', '欢', '<movie>', '的', '，', '推', '荐', '你', '看', '一', '去', '吧', '，', '无', '论', '今', '社会', '实', '现', '童话', '故事', '这', '个', '词', '是', '真', '的', '很', '不', '错', '的', '。', '[/s_response]', '一', '般', '的', '美', '式', '幽默', '和', '温暖', '。', '[/s_response]', '求', '了', '。', '[/s_response]', '。'], ['你', '看', '过', '<movie>', '吗', '，', '讽刺', '意', '味', '的', '美国', '电', '影', ' ', '。', '[/s_response]', '和', '悲', '伤', '的', '影像', '，', '故事', '节奏', '缓', '解', '读', '。', '[/s_response]', '起', '来', '也', '很', '不', '错', '。', '[/s_response]', '的', '一', '部', '讽刺', '片', '。', '[/s_response]', '剧', '。', '[/s_response]', '[/s_response]', '[/s_response]', '[/s_response]'], ['<movie>', '的', '风', '格', '。', '[/s_response]', '起', '来', '之', '中', '居', '然', '也', '说', '很', '有', '意', '思', '的', '一', '部', '电', '影', '。', '[/s_response]', '演', '技', '满', '分', '。', '[/s_response]', '开', '始', '。', '[/s_response]', '放', '弃', '生', '活', '化', '学', '拍', '成', '喜剧', '。', '[/s_response]', '求', '。', '[/s_response]', '。'], ['<movie>', '可', '以', '去', '看', '看', '，', '一', '部', '很', '温情', '的', '电', '影', '，', '世', '间', '万', '物', '的', '人性', '。', '[/s_response]', '生', '活', '的', '艰', '难', '。', '[/s_response]', '的', '谎', '言', '体', '会', '其', '实', '至', '极', '。', '[/s_response]', '好', '的', '影', '子', '。', '[/s_response]', '。', '[/s_response]', '。'], ['是', '啊', '，', '有', '时', '候', '心动', '的', '女', '生', '就', '可', '以', '满', '足', '你', '了', '。', '[/s_response]', '情绪', '心动', '的', '让', '人心', '动', '。', '[/s_response]', '上', '。', '[/s_response]', '里', '总', '想', '着', '怎', '么', '办', '？', '[/s_response]', '划', '，', '哈', '哈', '。', '[/s_response]', '。', '[/s_response]', '。', '[/s_response]', '。'], ['<movie>', '，', '良心', '推', '荐', '你', '去', '看', '看', '，', '<movie>', '，', '话', '说', '霍', '夫', '妇', '导演', '的', '顶', '级', '编剧', '，', '很', '有', '教育', '意', '义', '。', '[/s_response]', '战', '胜', '过', '。', '[/s_response]', '绝', '对', '精', '品', '。', '[/s_response]', '业', '啊', '。', '[/s_response]', '趣味', '。', '[/s_response]', '。', '[/s_response]'], ['吐', '血', '推', '荐', '<movie>', '，', ' ', '带', '有', '时代', '风', '格', '的', '黑帮', '电', '影', '化', '时代', '时代', '。', '[/s_response]', '战', '。', '[/s_response]', '的', '演', '绎', '。', '[/s_response]', '的', '你', '可', '以', '看', '一', '看', '。', '[/s_response]', '演', '的', '。', '[/s_response]', '俩', '女', '主', '演', '的', '[/s_response]', '。', '[/s_response]'], ['推', '荐', '你', '看', '<movie>', '，', '用', '艺术', '的', '手', '法', '展', '现', '了', '虚', '极', '强', '的', '人', '。', '[/s_response]', '却', '无', '比', '。', '[/s_response]', '和', '情感', '。', '[/s_response]', '才', '是', '最', '伟', '大', '的', '观', '点', '。', '[/s_response]', '才', '不', '一', '样', '。', '[/s_response]', '片', '。', '[/s_response]', '。'], ['你', '说', '的', '很', '对', '，', '给', '你', '推', '荐', '几', '部', '有', '关', '温情', '的', '电', '影', '看', '看', '呀', '？', '[/s_response]', '学', '的', '。', '[/s_response]', '了', '心情', '会', '有', '的', '。', '[/s_response]', '吗', '？', '[/s_response]', '要', '不', '要', '我', '给', '你', '推', '荐', '一', '下', '？', '[/s_response]', '？'], ['你', '看', '看', '<movie>', '吧', '，', '这', '部', '电', '影', '告', '诉', '我', '们', '每', '个', '人', '对', '人性', '中', '的', '思考', '。', '[/s_response]', '而', '且', '是', '反', '映', '了', '时代', '中', '的', '伟', '大', '。', '[/s_response]', '相', '识', '不', '同', '的', '独', '立', '故事', '。', '[/s_response]', '。', '[/s_response]', '。'], ['这', '种', '动画', '片', '我', '还', '是', '很', '喜', '欢', '的', '。', '[/s_response]', '这', '部', '动画', '电', '影', '也', '很', '适', '合', '你', '。', '[/s_response]', '要', '看', '动画', '电', '影', '来', '临', '时', '间', '。', '[/s_response]', '里', '他', '的', '那', '种', '喜', '欢', '。', '[/s_response]', '吗', '？', '[/s_response]', '。', '[/s_response]'], ['<movie>', '是', '非', '常', '好看', '的', '，', '讲', '述', '了', '意', '大', '利', '乡村', '的', '诞', '生', '，', '与', '国', '产', '阶级', '的', '隐喻', '。', '[/s_response]', '与', '现实', '主', '义', '。', '[/s_response]', '与', '现实', '主', '义', '的', '影片', '。', '[/s_response]', '能', 'g', 'e', 't', '到', '。', '[/s_response]', '。', '[/s_response]', '。']]
    # print(len(gth))
    # print(len(gen))
    pass

main()