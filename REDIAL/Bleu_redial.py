#coding=utf-8
import json
import sys
from nltk.translate import bleu_score as nltkbleu
import re
from option import option

mid2name = json.load(open('./dataset/mid2name_redial.json'))
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re_punc.sub(' ', text)
    def lower(text):
        return text.lower()
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
        bleu = nltkbleu.sentence_bleu(
            [normalize_answer(full_sen_gth).split(" ")],
            normalize_answer(full_sen_gen).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method7,
        )
        bleu_sum += bleu
        count += 1
    return bleu_sum/count