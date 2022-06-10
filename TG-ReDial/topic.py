import argparse
import random
from DataProcessor import DataSet
from get_logger  import get_logger
from get_logger  import task_uuid
from Vocab import  Vocab
from upcrtopic import Upcrtopic,Engine

main_logger = get_logger("main", './log/test.log')
main_logger.info("TASK ID {}".format(task_uuid))

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_data", choices=["test", "valid"], default="test")
    parser.add_argument("--super_rate", type=float, default=0.)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument('--inference',type=bool,default=False,help='')
    parser.add_argument("-use_cuda", "--use_cuda", type=bool, default=False)
    parser.add_argument("-gpu", "--gpu", type=str, default='1')
    parser.add_argument("--processed", type=bool, default=True, help='')
    args = parser.parse_args()
    return args

def main():
    random.seed(1234)
    args = config()
    main_logger.info("preparing data")
    dataset = DataSet(args=args)
    train, valid, test, users, user_cont = dataset.get_dialog(task='topic')
    vocab = Vocab()
    random.shuffle(train)
    excrs_topic = Upcrtopic(vocab=vocab,user_cont=user_cont)
    engine = Engine(model=excrs_topic,vocab=vocab)
    engine.train(train,test)

if __name__ == '__main__':
    main()