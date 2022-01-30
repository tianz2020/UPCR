#coding=utf-8

import argparse
import sys

import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import ipdb
import re
import jieba
from option import option



def bleu_cal(sen1, tar1):
    bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
    bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
    bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
    return bleu1, bleu2, bleu3, bleu4


def bleu(tokenized_gen, tokenized_tar):
    print_num = 0
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, count = 0, 0, 0, 0, 0
    for sen, tar in zip(tokenized_gen, tokenized_tar):

        for j,word in enumerate(sen):
            if word == option.EOS_RESPONSE:
                sen = sen[:j]
                break
        tar = tar[1:]
        for k,word in enumerate(tar):
            if word == option.EOS_RESPONSE:
                tar = tar[:k]
                break
        full_sen_gen = ''
        full_sen_gth = ''

        for word in sen:
            full_sen_gen += word

        for word in tar:
            full_sen_gth +=word

        if print_num<5:
            print("ground_truth:{}".format(full_sen_gth))
            print("generation:{}".format(full_sen_gen))
            sys.stdout.flush()
            print_num += 1

        sen_split_by_movie = list(full_sen_gen.split('<movie>'))
        sen_1 = []
        for i, sen_split in enumerate(sen_split_by_movie):
            for segment in jieba.cut(sen_split):
                sen_1.append(segment)
            if i != len(sen_split_by_movie) - 1:
                sen_1.append('<movie>')

        tar_split_by_movie = list(full_sen_gth.split('<movie>'))
        tar_1 = []
        for i, tar_split in enumerate(tar_split_by_movie):
            for segment in jieba.cut(tar_split):
                tar_1.append(segment)
            if i != len(tar_split_by_movie) - 1:
                tar_1.append('<movie>')


        bleu1, bleu2, bleu3, bleu4 = bleu_cal(sen_1, tar_1)
        bleu1_sum += bleu1
        bleu2_sum += bleu2
        bleu3_sum += bleu3
        bleu4_sum += bleu4
        count += 1

    return bleu1_sum / count, bleu2_sum / count, bleu3_sum / count, bleu4_sum / count
