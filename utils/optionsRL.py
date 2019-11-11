import sys
import argparse

def load_arguments():
	argparser = argparse.ArgumentParser(sys.argv[0])

	argparser.add_argument('--task',
		type = int,
		default = 1)
	argparser.add_argument('--embedding',
		type = str,
		default = "/home/yunshi/Word2vec/glove.840B.300d.zip",
		help = "path to pre-trained word vectors")
	argparser.add_argument('--save_id',
		type = str,
		default = "2",
		help = "save index")
	argparser.add_argument('--load_id',
		type = str,
		default = "1",
		help = "load index")
	argparser.add_argument('--train_q',
		type = str,
		default = "data/train_CQ",
		help = "path to training data")
	argparser.add_argument('--dev_q',
		type = str,
		default = "data/dev_CQ",
		help = "path to development data")
	argparser.add_argument('--test_q',
		type = str,
		default = "data/test_CQ",
		help = "path to test data")
	argparser.add_argument('--max_epoches',
		type = int,
		default = 100,
		help = "maximum of epoches")
	argparser.add_argument('--batch',
		type = int,
		default = 256,
		help = "mini-batch size")
	argparser.add_argument('--learning',
		type = str,
		default = "adam",
		help = "learning method")
	argparser.add_argument('--learning_rate',
		type = float,
		default = 0.0005,
		help = "learning rate")
	argparser.add_argument('--dropout',
		type = float,
		default = 0.2,
		help = "dropout probability")
	argparser.add_argument('--activation',
		type = str,
		default = "tanh",
		help = "type of activation function")
	argparser.add_argument('--hidden_dim',
		type = int,
		default = 150,
		help = "hidden dimension")
	argparser.add_argument('--is_fix_emb',
		type = int,
		default = 1)
	argparser.add_argument('--is_train',
		type = int,
		default = 1)
	argparser.add_argument('--only_eval',
		type = int,
		default = 0)
	argparser.add_argument('--max_plen',
		type = int,
		default = 30)
	argparser.add_argument('--max_qlen',
		type = int,
		default = 30)
	argparser.add_argument('--kernel_size',
		type = int,
		default =5)
	argparser.add_argument('--stride',
		type = int,
		default =1)
	argparser.add_argument('--padding',
		type = int,
		default =2)
	argparser.add_argument('--copy',
		type = int,
		default =1)

	args = argparser.parse_args()
	return args
