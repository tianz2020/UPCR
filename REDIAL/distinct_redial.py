from option import option
import json

mid2name = json.load(open('./dataset/mid2name_redial.json'))
def cal_calculate(outs):
    unigram_count = 0
    bigram_count = 0
    trigram_count = 0
    quagram_count = 0
    unigram_set = set()
    bigram_set = set()
    trigram_set = set()
    quagram_set = set()
    for sentence in outs:
        for j, word in enumerate(sentence):
            if word == option.EOS_RESPONSE:
                sentence = sentence[:j]
                break
        full_sen_gen = []
        for word in sentence:
            if '@' in word:
                movie_id = word[1:]
                if movie_id in mid2name:
                    movie = mid2name[movie_id]
                    tokens = movie.split(' ')
                    full_sen_gen.extend(tokens)
                else:
                    full_sen_gen.append(word)
            else:
                full_sen_gen.append(word)
        for word in full_sen_gen:
            unigram_count += 1
            unigram_set.add(word)
        for start in range(len(full_sen_gen) - 1):
            bg = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1])
            bigram_count += 1
            bigram_set.add(bg)
        for start in range(len(full_sen_gen) - 2):
            trg = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1]) + ' ' + str(full_sen_gen[start + 2])
            trigram_count += 1
            trigram_set.add(trg)
        for start in range(len(full_sen_gen) - 3):
            quag = str(full_sen_gen[start]) + ' ' + str(full_sen_gen[start + 1]) + ' ' + str(full_sen_gen[start + 2]) + ' ' + str(full_sen_gen[start + 3])
            quagram_count += 1
            quagram_set.add(quag)
    dis1 = len(unigram_set) / unigram_count
    dis2 = len(bigram_set) / bigram_count
    dis3 = len(trigram_set) / trigram_count
    dis4 = len(quagram_set) / quagram_count
    return dis1, dis2, dis3, dis4