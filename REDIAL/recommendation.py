import argparse
from DataProcessor_Redial import DataSet
from get_logger  import get_logger
from get_logger  import task_uuid
from VocabRedial import  Vocab
from upcrRecRedial import UPCR,Engine

main_logger = get_logger("main", './log/test.log')
main_logger.info("TASK ID {}".format(task_uuid))


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference',type=bool,default=False,help='')
    parser.add_argument("-use_cuda", "--use_cuda", type=bool, default=False)
    parser.add_argument("-gpu", "--gpu", type=str, default='1')
    parser.add_argument("--processed", type=bool, default=True, help='')
    args = parser.parse_args()
    return args

def main():
    args = config()
    main_logger.info("preparing data")
    dataset = DataSet(args=args)
    train, valid, test, users, user_cont = dataset.get_dialog('rec')
    vocab = Vocab()
    excrs_topic = UPCR(vocab=vocab,user_cont=user_cont)
    engine = Engine(model=excrs_topic,vocab=vocab)
    engine.train(train,valid,test)

if __name__ == '__main__':
    main()