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
        tar = tar[1:] 
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
