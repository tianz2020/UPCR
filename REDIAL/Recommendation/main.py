import argparse
import json
import random
import ipdb
from DataProcessor_Redial import DataSet
from get_logger  import get_logger
from get_logger  import task_uuid
from VocabRedial import  Vocab
from excrsRecRedial import ExcrsTopic,EngineTopic
from DataLoaderRecRedial import  DataLoaderRec
import pickle
from  tqdm import  tqdm

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

    # with open('./dataset/train_rec_redial_dbpedia.pkl', 'wb+') as f:
    #    pickle.dump(train,f)
    #
    # with open('./dataset/valid_rec_redial_dbpedia.pkl', 'wb+') as f1:
    #    pickle.dump(valid,f1)
    #
    # with open('./dataset/test_rec_redial_dbpedia.pkl', 'wb+') as f2:
    #    pickle.dump(test,f2)


    # dataloader = DataLoaderRec(train,vocab)
    # for d in tqdm(dataloader):
    #     pass
    #
    # print(dataloader.num)
    # print(dataloader.num_1)

    train = train + valid
    random.shuffle(train)
    excrs_topic = ExcrsTopic(vocab=vocab,user_cont=user_cont)
    engine = EngineTopic(model=excrs_topic,vocab=vocab)
    # engine.test(test,'test')
    engine.train(train,test)

if __name__ == '__main__':
    main()