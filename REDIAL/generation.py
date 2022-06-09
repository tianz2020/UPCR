import argparse
import random
from DataProcessor_Redial import DataSet
from get_logger  import get_logger
from get_logger  import task_uuid
from VocabRedial import  Vocab
from upcrRespRedial import UPCR,Engine

main_logger = get_logger("main", './log/test.log')
main_logger.info("TASK ID {}".format(task_uuid))

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference',type=bool,default=False,help='训练还是测试')
    parser.add_argument("-use_cuda", "--use_cuda", type=bool, default=False)
    parser.add_argument("-gpu", "--gpu", type=str, default='1')
    parser.add_argument("--processed", type=bool, default=True, help='数据是否已经预处理')
    args = parser.parse_args()
    return args

def main():
    random.seed(1234)
    args = config()
    main_logger.info("preparing data")
    dataset = DataSet(args=args)
    train, valid, test, users, user_cont = dataset.get_dialog("gene")
    vocab = Vocab()
    random.shuffle(train)
    excrs_topic = UPCR(vocab=vocab,user_cont=user_cont)
    engine = Engine(model=excrs_topic,vocab=vocab)
    engine.train(train,valid,test)

if __name__ == '__main__':
    main()