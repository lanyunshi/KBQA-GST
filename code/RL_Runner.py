from utils.utils import *
import numpy as np
from time import gmtime, strftime
import os
import pickle
import re
from utils.optionsRL import load_arguments

from Dataset import Dataset
from Models import Train_loop

def read_dic(data_path):
    dic = {}
    with open(data_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            d, idx = line.strip('\n').split('\t')
            dic[d] = int(idx)
    return dic

def main(args, load_save = None, load_id = None):
    print("Train and test for task ...")
    print('cuda is available', torch.cuda.is_available())
    if load_save:
        outfile = load_save
        args = pickle.load(open('%s/config%s.pkl' %(outfile, load_id), 'rb'))
        dic = pickle.load(open('%s/dic.pkl' %outfile, 'rb')) #, encoding = 'latin1'
        #dic = read_dic('%s/dic.txt' %outfile)
    else:
        outfile = 'saved-model/' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + 'RL'
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        dic = {"<PAD>": 0, "<UNK>": 1, "<E>": 2}

    args.model_type = 'TG+EE'
    args.train_q = 'data/train_WBQ'
    args.dev_q = 'data/dev_WBQ'
    args.test_q = 'data/test_WBQ'
    args.tp_model = 'CoAttention'
    args.model = 'CNN'
    args.is_train = 1
    args.save_id = 'TG+EE_tmp'
    args.load_id = 'TG+EE'
    args.train_execution_engine = 1
    args.train_program_generator = 0
    args.batch = 10
    args.display = 50
    args.only_eval = 0
    args.learning_rate = 1e-05
    args.dropout = 0.
    args.use_gpu = 2
    args.hidden_dim = 200
    args.max_epoches = 1000
    args.use_units = 1
    args.embedding = '/home/yunshi/Word2vec/glove.840B.300d.zip' #'/opt/shared/Word2vec/glove.840B.300d.zip' #

    if load_save:
        if args.is_train:
            train_data = Dataset(args.train_q, is_train= True, use_units = args.use_units)
            dev_data = Dataset(args.dev_q, is_train= True, use_units = args.use_units)
            train_data.concate(dev_data)
            test_data = Dataset(args.test_q, use_units = args.use_units)
        else:
            test_data = Dataset(args.test_q, use_units = args.use_units)
            train_data = test_data
            dev_data = test_data

    else:
        train_data = Dataset(args.train_q, dic) #, limit = 1000
        dev_data = Dataset(args.dev_q, dic)#, limit = 1000
        test_data = Dataset(args.test_q, dic)#, limit = 500
        pickle.dump(dic, open('%s/dic.pkl' %outfile, 'wb'), protocol = 2)

    print('train: %s dev: %s test: %s' %(train_data.len(), dev_data.len(), test_data.len()))
    if load_save and os.path.exists('%s/weights%s.npy' %(outfile, args.load_id)):
        print('Load pre-trained Embedding ...')
        emb0, emb1 = np.load('%s/weights%s.npy' %(outfile, args.load_id))
        emb = [emb0, emb1]
    else:
        emb = initialize_vocab(dic, args.embedding)
        #emb1 = np.load('%s/weightsEE2.npy' %(outfile))[0]
        emb = [emb, emb]
        np.save('%s/weights%s' %(outfile, args.save_id), emb)

    pickle.dump(args, open('%s/config%s.pkl' %(outfile, args.save_id), 'wb'), protocol = 2)
    save_config(args, '%s/config%s.txt' %(outfile, args.save_id))

    print('Parser Arguments')
    for key, value in args.__dict__.items():
        print('{0}: {1}'.format(key, value))

    train_loop = Train_loop(args = args, emb = emb)

    train_loop.train(train_data, dev_data, test_data, dic, outfile)

if __name__ == '__main__':
    args = load_arguments()

    if args.task == 1:
        main(args, load_save = 'saved-model/WBQ', load_id = 'TG+EE')
    elif args.task == 2:
        main(args, load_save = 'saved-model/SQ', load_id = 'TG+EE')
    elif args.task == 3:
        main(args, load_save = 'saved-model/CWQ', load_id = 'TG+EE')
    else:
        main(args)

'''
python code/RL_Runner.py --task 1
'''
