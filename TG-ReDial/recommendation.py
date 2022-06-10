import argparse
import random
from DataProcessor import DataSet
from get_logger  import get_logger
from get_logger  import task_uuid
from Vocab import  Vocab
from upcrrec import Upcrrec,Engine

main_logger = get_logger("main", './log/test.log')
main_logger.info("TASK ID {}".format(task_uuid))

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference',type=bool,default=False,help='')
    parser.add_argument("--processed", type=bool, default=True, help='')
    args = parser.parse_args()
    return args

def main():
    random.seed(1234)
    args = config()
    main_logger.info("preparing data")
    dataset = DataSet(args=args)
    train, valid, test, users, user_cont = dataset.get_dialog(task='rec')
    vocab = Vocab(task='rec')
    random.shuffle(train)
    excrs = Upcrrec(vocab=vocab,user_cont=user_cont)
    engine = Engine(model=excrs,vocab=vocab)
    engine.train(train,test)

if __name__ == '__main__':
    main()