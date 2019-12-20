from __future__ import division
from __future__ import print_function

import os
import zipfile
import re
import numpy as np
import copy
import time

import torch

def softmax(x):
    x = x - np.max(x, axis = 1).reshape((-1, 1))
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis = 1).reshape((-1, 1))
    return softmax_x

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def initialize_vocab(dic, path):
    vocab = np.random.uniform(-0.1, 0.1, (len(dic), 300))
    seen = 0

    gloves = zipfile.ZipFile(path)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0].decode('utf-8')
                    embedding = splitline[1:]
                    if word in dic and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        vocab[dic[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen += 1
    vocab = vocab.astype(np.float32)
    vocab[0, :] = 0.
    print('pretrianed vocab %s among %s' %(seen, len(dic)))
    return vocab

def save_config(config, path):
    with open(path, 'w') as f:
        for key, value in config.__dict__.items():
            f.write('{0}: {1}\n'.format(key, value))

def create_batches(N, batch_size, skip_idx = None, is_shuffle = False):
    batches = []
    shuffle_batch = np.arange(N)
    if skip_idx:
        shuffle_batch = list(set(shuffle_batch) - set(skip_idx))
    if is_shuffle:
        np.random.shuffle(shuffle_batch)
    M = int(np.ceil(N*1./batch_size))
    for i in range(M):
        batches += [shuffle_batch[batch_size*i: (batch_size)*(i+1)]]
    return batches

def sents2index(sents, dic):
    return [dic[w] if w in dic else dic['<UNK>'] for w in sents.split(' ')]

def words2index(words, dic):
    words = re.sub('[^a-z]', '', words)
    return [dic[w] for w in words]

def index2sents(index, dic2):
    return ' '.join([dic2[i] for i in index if i != 0])

def generate_predicate(cand_path):
    rels = []
    for c in cand_path:
        if len(c.split('\t')) == 2:
            x0, r1 = c.split('\t')
            if not re.search('^[mg]\.', r1):
                rs = [r1]
            else:
                print(c)
                exit()
        elif len(c.split('\t')) == 3:
            x0, r1, r2 = c.split('\t')
            if not (re.search('^[mg]\.', r1) or re.search('^[mg]\.', r2)):
                rs = [r1, r2]
            else:
                print(c)
                exit()
        elif len(c.split('\t')) == 4:
            if re.search('^[mg]\.', c.split('\t')[2]):
                x1, y1, x0, y2 = c.split('\t')
            else:
                x1, y1, y2, x0 = c.split('\t')
            if not (re.search('^[mg]\.', y1) or re.search('^[mg]\.', y2) or re.search('^[mg]\.', x0)):
                rs = [y1, x0, y2]
            elif not (re.search('^[mg]\.', y1) or re.search('^[mg]\.', y2)):
                rs = [y1, y2]
            else:
                print(c)
                exit()
        elif len(c.split('\t')) == 5:
            if re.search('^[mg]\.', c.split('\t')[3]):
                x1, y1, y2, x0, y0 = c.split('\t')
            else:
                x1, y1, y2, y0, x0 = c.split('\t')
            rs = [y1, y2, y0]
        else:
            print(c)
            exit()
        rel = [r.split('.')[-1].split('_') for r in rs] #
        rel = ' '.join(sum(rel, []))
        rels += [rel]
    return rels

def generate_relation(cand_path, to_entity):
    rels = []
    topics = []
    for c in cand_path:
        rel = []
        topic = []
        for w in c.split('\t'):
            if re.search('^[mg]\.', w):
                if w in to_entity:
                    rel += to_entity[w][0].split(' ')
                    topic += [to_entity[w][1]]
                else:
                    rel += ['<UNK>']
            else:
                rel += '_'.join(w.split('.')[-2:]).split('_') #w.split('.')[-1].split('_')
        rel = ' '.join(rel)
        rels += [rel]
        topics += [topic]
    return topics, rels

def generate_types(cand_path_cont):
    types = []
    for ts in cand_path_cont:
        type = []
        for t in ts:
            for w in re.split('\.|\_| |\/', t):
                type += [w]
        types += [' '.join(type)]
    return types

def obtain_topic_entity_candidate_path_features(data, batches, dic, is_EE = False):
    dic, dic2 = dic
    t = np.zeros((len(batches), 100), dtype = np.int32)
    q = np.zeros((len(batches), 100), dtype = np.int32)
    if is_EE:
        p = np.zeros((len(batches), 2000, 100), dtype = np.int32)
        r = np.zeros((len(batches), 2000, 100), dtype = np.int32)
        y = np.zeros((len(batches), 2000, 100), dtype = np.int32)
        an = np.zeros((len(batches), 2000), dtype = np.float32)
        anlen = np.zeros((len(batches), 2000), dtype = np.int32)
        l = np.zeros((len(batches), 2000), dtype = np.float32)
        orig_l = np.zeros((len(batches), 2000), dtype = np.float32)
        max_cand, max_tlen, max_qlen, max_plen, max_rlen, max_ylen = 1, 1, 1, 1, 1, 1
    else:
        els = np.zeros((len(batches), 500), dtype = np.float32)
        elo = np.zeros((len(batches), 500), dtype = np.float32)
        cate = np.zeros((len(batches), 500, 2), dtype = np.int32)
        alia = np.zeros((len(batches), 500, 100), dtype = np.int32)
        el = np.zeros((len(batches), 500), dtype = np.int32)
        orig_l = np.zeros((len(batches), 500), dtype = np.float32)
        topic_entities = []
        answers = []
        to_entities, to_entities2 = [], []
        candidate_paths_pool = []
        max_el, max_tlen, max_qlen, max_alen = 1, 1, 1, 1

    for i, b in enumerate(batches):
        sentence, template, answer, cand_path, cand_path_score, cand_path_an, cand_path_cont, to_entity = data[b]
        input = []

        sent_idx = sents2index(sentence, dic)
        q[i, :len(sent_idx)] = sent_idx
        if len(sent_idx) > max_qlen:
            max_qlen = len(sent_idx)
        tem_idx = sents2index(template, dic)
        t[i, :len(tem_idx)] = tem_idx
        if len(tem_idx) > max_tlen:
            max_tlen = len(tem_idx)

        if is_EE:
            predicate = generate_predicate(cand_path)
            for p_idx, _p in enumerate(predicate):
                predi_idx = sents2index(_p, dic)
                p[i, p_idx, :len(predi_idx)] = predi_idx
                if len(predi_idx) > max_plen:
                    max_plen = len(predi_idx)
            if len(predicate) > max_cand:
                max_cand = len(predicate)

            _, relation = generate_relation(cand_path, to_entity)
            for r_idx, _r in enumerate(relation):
                rel_idx = sents2index(_r, dic)[:100]
                r[i, r_idx, :len(rel_idx)] = rel_idx
                if len(rel_idx) > max_rlen:
                    max_rlen = len(rel_idx)

            types = generate_types(cand_path_cont)
            for y_idx, _y in enumerate(types):
                type_idx = sents2index(_y, dic)[:100]
                y[i, y_idx, :len(type_idx)] = type_idx
                if len(type_idx) > max_ylen:
                    max_ylen = len(type_idx)

            for a_idx, _a in enumerate(cand_path_an):
                anlen[i, a_idx] = _a
                a_split = cand_path[a_idx].split('\t')
                w_idxes = np.arange(len(a_split))
                an[i, a_idx] += np.sum([float(to_entity[a_split[w_idx]][1]) for w_idx in w_idxes if (re.search('^[mg]\.', a_split[w_idx])
                                and a_split[w_idx] in to_entity)])

                l[i, a_idx] = cand_path_score[a_idx]
            orig_l[i, :] = copy.deepcopy(l[i, :])
            if np.sum(l[i, :]) == 0:
                l[i, :] = 1.

        else:
            topic_entity = []
            for m_idx, mid in enumerate(to_entity):
                if re.search('^[mg]\.', mid):
                    ali = to_entity[mid][0]
                    #elo[i, m_idx] = to_entity[mid][1]
                    els[i, m_idx] = to_entity[mid][2]
                    topic_entity += [mid]
                else:
                    ali = ' '.join(re.split('\.|\_', mid.lower())[1: ])
                    elo[i, m_idx] = to_entity[mid][1]
                    if to_entity[mid][0] == 1:
                        cate[i, m_idx, 0] = 1
                    else:
                        cate[i, m_idx, 1] = 1
                    topic_entity += [mid]
                ali_idx = sents2index(ali, dic)[:50]
                try:
                    alia[i, m_idx, :len(ali_idx)] = ali_idx
                except:
                    print(b); print(ali); exit()
                if len(ali_idx) > max_alen:
                    max_alen = len(ali_idx)
                el[i, m_idx] = to_entity[mid][2]
            orig_l[i, :] = copy.deepcopy(el[i, :])
            # if np.sum(el[i, :]) == 0:
            #     el[i, :] = 1.
            if len(to_entity) > max_el:
                max_el = len(to_entity)
            topic_entities += [topic_entity]
            answers += [answer]
            to_entities += [(cand_path, cand_path_score, cand_path_an, cand_path_cont)]
            to_entities2 += [to_entity]

    if is_EE:
        max_cand = np.min([max_cand, 1000])
        q = q[:, :max_qlen]
        t = t[:, :max_tlen]
        p = p[:, :max_cand, :max_plen]
        r = r[:, :max_cand, :max_rlen]
        y = y[:, :max_cand, :max_ylen]
        an = an[:, :max_cand]
        anlen = anlen[:, :max_cand]
        l = l[:, :max_cand]
        l = l/np.maximum(np.sum(l, axis = 1).reshape((-1, 1)), 1.e-10)
        return t, q, p, r, y, an, anlen, orig_l, l, 0
    else:
        els = els[:, :max_el]
        elo = elo[:, :max_el]
        cate = cate[:, :max_el, :]
        q = q[:, :max_qlen]
        t = t[:, :max_tlen]
        alia = alia[:, :max_el, :max_alen]
        el = el[:, :max_el]
        orig_l = orig_l[:, :max_el]
        #el = el/np.sum(el, axis = 1).reshape((-1, 1))
        return t, q, els, elo, cate, alia, orig_l, el, (topic_entities, answers, to_entities, to_entities2)

def obtain_candidate_path_features(to_entities, cand_paths, dic, te=None):
    dic, dic2 = dic
    p = np.zeros((len(cand_paths), 5000, 100), dtype = np.int32)
    r = np.zeros((len(cand_paths), 5000, 100), dtype = np.int32)
    y = np.zeros((len(cand_paths), 5000, 100), dtype = np.int32)
    an = np.zeros((len(cand_paths), 5000), dtype = np.float32)
    anlen = np.zeros((len(cand_paths), 5000), dtype = np.int32)
    l = np.zeros((len(cand_paths), 5000), dtype = np.float32)
    orig_l = np.zeros((len(cand_paths), 5000), dtype = np.float32)
    max_cand, max_tlen, max_qlen, max_plen, max_rlen, max_ylen = 1, 1, 1, 1, 1, 1

    for i in range(len(cand_paths)):
        #print(to_entities[i])
        predicate = generate_predicate(cand_paths[i])
        for p_idx, _p in enumerate(predicate):
            predi_idx = sents2index(_p, dic)
            p[i, p_idx, :len(predi_idx)] = predi_idx
            if len(predi_idx) > max_plen:
                max_plen = len(predi_idx)
        if len(predicate) > max_cand:
            max_cand = len(predicate)

        _, relation = generate_relation(cand_paths[i], to_entities[i])
        for r_idx, _r in enumerate(relation):
            rel_idx = sents2index(_r, dic)
            r[i, r_idx, :len(rel_idx)] = rel_idx
            if len(rel_idx) > max_rlen:
                max_rlen = len(rel_idx)

        types = generate_types([cand_paths[i][_a][2] for _a in cand_paths[i]])
        for y_idx, _y in enumerate(types):
            type_idx = sents2index(_y, dic)[:100]
            y[i, y_idx, :len(type_idx)] = type_idx
            if len(type_idx) > max_ylen:
                max_ylen = len(type_idx)

        #print(te)
        for a_idx, _a in enumerate(cand_paths[i]):
            time1 = time.time()
            a_split = _a.split('\t')
            w_idxes = np.arange(len(a_split))
            if te is not None:
                te_no_prefix = dict([(re.sub('^[12]', '', t), te[i][t]) for t in te[i]])
                tmp = [np.max([float(te_no_prefix[a_split[w_idx]]), 1e-10]) for w_idx in w_idxes if a_split[w_idx] in te_no_prefix]
                tmp = np.sum(tmp) if len(tmp) > 0 else 0
                an[i, a_idx] += tmp
            else:
                an[i, a_idx] += np.sum([float(to_entities[i][a_split[w_idx]][1]) for w_idx in w_idxes if a_split[w_idx] in to_entities[i]])#cand_paths[i][_a][0]
            # if a_idx < 10:
            #     print('%s\t%s\t%s\t%s' %(list(cand_paths[i].keys())[a_idx], relation[a_idx], types[a_idx], an[i, a_idx]))
            anlen[i, a_idx] = cand_paths[i][_a][0]
            f1 = cand_paths[i][_a][1]
            l[i, a_idx] = f1

        orig_l[i, :] = copy.deepcopy(l[i, :])
        if np.sum(l[i, :]) == 0:
            l[i, :] = 1.

    p = p[:, :max_cand, :max_plen]
    r = r[:, :max_cand, :max_rlen]
    y = y[:, :max_cand, :max_ylen]
    an = an[:, :max_cand]
    anlen = anlen[:, :max_cand]
    l = l[:, :max_cand]
    l = l/np.sum(l, axis = 1).reshape((-1, 1))
    return p, r, y, an, anlen, orig_l, l

def get_topic_entity_via_topk(topk, topic_entities, cate, topkscore, cand_path):

    def has_graph(subset, bigset, bigtypeset):
        subset = set([s[1:] if not re.search('^[mg]\.', s) else s for s in subset])
        for c_idx, c in enumerate(bigset):
            if subset.issubset(set(c.split('\t') + bigtypeset[c_idx])):
                return True
        return False

    #topkp = softmax(topkscore)
    tes = []
    for t_idx in range(len(topic_entities)):
        te = {}
        accumscore = 0
        #print(topk[t_idx])
        for k_idx, k in enumerate(topk[t_idx]):
            prefix = ''
            if cate[t_idx][k][0] == 1:
                prefix = '1'
            elif cate[t_idx][k][1] == 1:
                prefix = '2'
            if k < len(topic_entities[t_idx]):
                if accumscore < 0.9999: #len(te) < 3:#has_graph(set(list(:question.keys()) + [prefix + topic_entities[t_idx][k]]), cand_path[t_idx][0], cand_path[t_idx][3]) \
                    #and ():
                    try:
                        te[prefix + topic_entities[t_idx][k]] = topkscore[t_idx][k_idx]
                        accumscore += topkscore[t_idx][k_idx]
                    except:
                        pass
        tes += [te]
    return tes

def print_pred(preds, shuffle_batch):
    shuffle_batch = np.concatenate(shuffle_batch)
    idx = sorted(range(len(shuffle_batch)), key = lambda x: shuffle_batch[x])
    pred_text = []
    for i in range(len(idx)):
        if idx[i] < len(preds):
            pred_text += ['%s\t%s' %(shuffle_batch[idx[i]]+1, preds[idx[i]])]
    return pred_text

def save_pred(file, preds):
    with open(file, 'w') as f:
        f.write('\n'.join(preds).encode('ascii', 'ignore').decode('ascii'))
    f.close()
