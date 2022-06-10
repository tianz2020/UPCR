import jieba
from  option import  option
def cal_calculate(tokenized_gen, tokenized_tar):
    dis1=0
    dis_set1=set()
    dis2=0
    dis_set2=set()
    for sen, tar in zip(tokenized_gen, tokenized_tar):
        for j,word in enumerate(sen):
            if word == option.EOS_RESPONSE:
                sen = sen[:j]
                break
        full_sen_gen = ''
        for word in sen:
            full_sen_gen += word
        sen_split_by_movie = list(full_sen_gen.split('<movie>'))
        sen_1 = []
        for i, sen_split in enumerate(sen_split_by_movie):
            for segment in jieba.cut(sen_split):
                sen_1.append(segment)
            if i != len(sen_split_by_movie) - 1:
                sen_1.append('<movie>')
        prediction = sen_1
        for word in prediction:
            dis_set1.add(word)
            dis1 += 1
        for i in range(1, len(prediction)):
            dis_set2.add(prediction[i - 1] + ' ' + prediction[i])
            dis2 += 1
    return len(dis_set1)/dis1, len(dis_set2)/dis2