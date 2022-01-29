import argparse
import json
import random
from DataProcessor import DataSet
from get_logger  import get_logger
from get_logger  import task_uuid
# from DataLoader1 import DataLoader
from Vocab import  Vocab
from excrs_allvocab import Excrs_allvocab,Engine_allvocab
from random import  shuffle


main_logger = get_logger("main", './log/test.log')
main_logger.info("TASK ID {}".format(task_uuid))


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", choices=["meddg", "meddg_dev", "meddialog", "kamed"], default="meddg_dev")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_data", choices=["test", "valid"], default="test")

    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="train batch size")
    parser.add_argument("-sbs", "--stack_batch_size", type=int, help="batch size after gradient stack.")
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=32, help="test batch size")

    parser.add_argument("--worker_num", type=int, default=5)
    parser.add_argument("--super_rate", type=float, default=0.)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--ckpt", type=str, default=None)

    # Ablation
    parser.add_argument("--super_only", action="store_true")
    parser.add_argument("--wos", action="store_true", help="without state")
    parser.add_argument("--woz", action="store_true", help="without z")
    parser.add_argument("--woa", action="store_true", help="without action")
    parser.add_argument("--wo_er", action="store_true", help="without entropy restrain regularization")
    parser.add_argument("--wo_rp", action="store_true", help="without repeat penalty regularization")
    parser.add_argument("--wo_rl", action="store_true", help="without reinforcement learning")
    parser.add_argument("--hungary", action="store_true")

    # train epochs
    parser.add_argument("--s_train", type=int, default=2, help="state train epoch number")
    parser.add_argument("--a_train", type=int, default=40, help="action train epoch number")
    parser.add_argument("--rl_train", type=int, default=5, help="rl train epoch number")

    # hyper-lambda
    parser.add_argument("--scl", type=float, default=5.0)  # state copy lambda
    parser.add_argument("--acl", type=float, default=5.0)

    parser.add_argument('--log_path',default=r'C:\Users\Administrator\Desktop\mywork_log\{}.log',type=str,required=False,help='训练日志存放位置')
    parser.add_argument('--inference',type=bool,default=False,help='训练还是测试')
    parser.add_argument("-epoch", "--epoch", type=int, default=500)
    parser.add_argument("-use_cuda", "--use_cuda", type=bool, default=False)
    parser.add_argument("-gpu", "--gpu", type=str, default='1')
    parser.add_argument("-vocab_path","--vocab_path",type=str,default=r"C:\Users\Administrator\Desktop\TG_data\vocab.txt",help='用于初始化分词器的字典')
    parser.add_argument("--processed", type=bool, default=True, help='数据是否已经预处理')
    args = parser.parse_args()
    #option_update(args)
    #log_config()
    return args


def main():
    random.seed(1234)
    args = config()
    # 数据预处理
    main_logger.info("preparing data")
    dataset = DataSet(args=args)
    train, valid, test, users, user_cont = dataset.get_dialog()
    vocab = Vocab()

    # with open('./dataset/train_movie_1127.pkl', 'wb+') as f:
    #    pickle.dump(train,f)

    # with open('./dataset/valid_movie_1127.pkl', 'wb+') as f1:
    #    pickle.dump(valid,f1)

    # with open('./dataset/test_movie_1127.pkl', 'wb+') as f2:
    #    pickle.dump(test,f2)

    # all = train + test
    # dataloader = DataLoaderRec(test,vocab)
    # for d in dataloader:
    #     pass
    #
    # print(dataloader.num)
    # print(dataloader.num_1)

    topic_graph = json.load(open('./dataset/graph_rec_full.json'))
    # graph = json.load(open('./dataset/topic2movie.json'))
    # for topic,relations in tqdm(topic_graph.items()):
    #     origin = graph[topic]
    #     for t in origin:
    #         if t not in relations:
    #             topic_graph[topic] = topic_graph[topic] + [t]


    for topic,relations in topic_graph.items():
        shuffle(topic_graph[topic][0:400])

    json.dump(topic_graph,open('./dataset/graph_rec_full_1.json','w+'))

    random.shuffle(train)
    excrs = Excrs_allvocab(vocab=vocab,user_cont=user_cont)
    engine = Engine_allvocab(model=excrs,vocab=vocab)
    # engine.test(test,'test')
    engine.train(train,test)


if __name__ == '__main__':
    main()