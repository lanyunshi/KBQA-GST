import torch
from utils.utils import *
import torch.nn as nn
from GetExecutionEngine import Ranker
from GetTopicGenerator import TopicGenerator
from utils.GenerateCandidatePathViaSparqlPool import generate_candidate_path_via_sparql_pool
from torch.autograd import Variable
import GenerateFinalPredictions
import math

torch.set_num_threads(7)
try:
    import sys
    sys.path.append('/stage/allennlp')
    from allennlp.commands.elmo import ElmoEmbedder
except:
    pass

eps = np.finfo(np.float32).eps.item()

def get_weights_and_biases(model, save_para):
    state_dict = {}
    old_dict = model.state_dict()
    for var in old_dict:
        if 'emb_init' not in var and var in save_para:
            state_dict[var] = old_dict[var]
    return state_dict

class Model(nn.Module):

    def __init__(self, args, emb):
        super(Model, self).__init__()
        self.args = args
        self.emb = emb
        if args.model_type == 'TG' or args.model_type == 'TG+EE':
            self.topic_generator = TopicGenerator(args, emb[0])
            self.tp_optimizer = torch.optim.Adam(self.topic_generator.parameters(),
                                        lr = args.learning_rate)
            #self.topic_generator.emb_init.weight.requires_grad = False
            print('Here is the topic entity generator:')
            print(self.topic_generator)
        if args.model_type == 'EE' or args.model_type == 'TG+EE':
            self.execution_engine = Ranker(args, emb[1])
            self.ee_optimizer = torch.optim.Adam(self.execution_engine.parameters(),
                                        lr = args.learning_rate)
            #self.execution_engine.emb_init.weight.requires_grad = False
            print('Here is the execution engine:')
            print(self.execution_engine)
        if args.model_type == 'TG+EE':
            pass#self.execution_engine.emb_init = self.topic_generator.emb_init

    def TG_forward(self, q, els, elo, cate, alia, orig_l, el, dic):
        dic, dic2 = dic
        self.topic_generator.train()
        if self.args.tp_model == 'CHARLSTM':
            q, q_char = q
            alia, alia_char = alia
            _q_char = Variable(torch.LongTensor(q_char), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q_char), requires_grad = False)
            _alia_char = Variable(torch.LongTensor(alia_char), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia_char), requires_grad = False)
        _els = Variable(torch.FloatTensor(els), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(els), requires_grad = False)
        _elo = Variable(torch.FloatTensor(elo), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(elo), requires_grad = False)
        _cate = Variable(torch.FloatTensor(cate), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(cate), requires_grad = False)
        _q = Variable(torch.LongTensor(q), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q), requires_grad = False)
        _alia = Variable(torch.LongTensor(alia), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia), requires_grad = False)
        _el = Variable(torch.FloatTensor(el), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(el), requires_grad = False)
        self.tp_optimizer.zero_grad()

        if self.args.tp_model == 'CHARLSTM':
            _logits, _ = self.topic_generator(_q, _els, _elo, _cate, _alia, q_char=_q_char, alia_char = _alia_char)
        else:
            _logits, _ = self.topic_generator(_q, _els, _elo, _cate, _alia)
        _loss = self.topic_generator.obtain_loss(_logits, _el)

        _loss.backward()
        self.tp_optimizer.step()
        loss = _loss.cpu().data.numpy() if self.args.use_gpu else _loss.data.numpy()
        logits = _logits.cpu().data.numpy() if self.args.use_gpu else _logits.data.numpy()
        prob = sigmoid(logits) > 0.5
        evals = np.sum(orig_l * prob, 1)/np.maximum(np.sum(orig_l, 1), 1e-10)
        # prob = softmax(logits)
        prob = np.argmax(logits, axis = 1)
        # evals = orig_l[np.arange(prob.shape[0]), prob]
        return loss, evals, prob, els+elo, orig_l

    def EE_forward(self, t, q, p, r, y, an, anlen, orig_l, l, elmos, dic):
        dic, dic2 = dic
        self.execution_engine.train()
        _t = Variable(torch.LongTensor(t), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(t), requires_grad = False)
        _q = Variable(torch.LongTensor(q), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q), requires_grad = False)
        _p = Variable(torch.LongTensor(p), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(p), requires_grad = False)
        _r = Variable(torch.LongTensor(r), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(r), requires_grad = False)
        _y = Variable(torch.LongTensor(y), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(y), requires_grad = False)
        _an = Variable(torch.FloatTensor(an), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(an), requires_grad = False)
        _anlen = Variable(torch.FloatTensor(anlen), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(anlen), requires_grad = False)
        _l = Variable(torch.FloatTensor(l), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(l), requires_grad = False)
        self.ee_optimizer.zero_grad()

        _logit, _prob = self.execution_engine(_t, _q, _p, _r, _y, _an, _anlen, elmos = elmos)
        _loss = self.execution_engine.obtain_loss(_prob, _l)
        _loss.backward()
        self.ee_optimizer.step()

        loss = _loss.cpu().data.numpy() if self.args.use_gpu else _loss.data.numpy()
        prob = _prob.cpu().data.numpy() if self.args.use_gpu else _prob.data.numpy()
        prob = np.argmax(prob, axis = 1)
        evals = orig_l[np.arange(prob.shape[0]), prob]
        return loss, evals, prob, r

    def TGEE_forward(self, t, els, elo, cate, q, alia, orig_l, el, others, coincident, dic):
        dic, dic2 = dic
        topic_entities, answers, to_entities, to_entities2 = others
        if self.args.tp_model == 'CHARLSTM':
            q, q_char = q
            alia, alia_char = alia
            _q_char = Variable(torch.LongTensor(q_char), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q_char), requires_grad = False)
            _alia_char = Variable(torch.LongTensor(alia_char), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia_char), requires_grad = False)
        self.topic_generator.train()
        self.execution_engine.train()
        _els = Variable(torch.FloatTensor(els), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(els), requires_grad = False)
        _elo = Variable(torch.FloatTensor(elo), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(elo), requires_grad = False)
        _cate = Variable(torch.FloatTensor(cate), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(cate), requires_grad = False)
        _q = Variable(torch.LongTensor(q), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q), requires_grad = False)
        _alia = Variable(torch.LongTensor(alia), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia), requires_grad = False)
        _el = Variable(torch.LongTensor(el), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(el), requires_grad = False)

        if self.args.tp_model == 'CHARLSTM':
            _logits, _atten = self.topic_generator(_q, _els, _elo, _cate, _alia, q_char = _q_char, alia_char = _alia_char)
        else:
            _logits, _atten = self.topic_generator(_q, _els, _elo, _cate, _alia)
        _topk, _prob, _attn = self.topic_generator.reinforce_sample(_logits, topic_entities, alia, k = 3, atten = None) #_atten
        #_topk, _prob, _attn = self.topic_generator.sample(_logits, topic_entities, alia, k = 3) # _logits
        topk = _topk.cpu().data.numpy() if self.args.use_gpu else _topk.data.numpy()
        prob = _prob.cpu().data.numpy() if self.args.use_gpu else _prob.data.numpy()
        attn = _attn.cpu().data.numpy() if self.args.use_gpu else _attn.data.numpy()
        te = get_topic_entity_via_topk(topk, topic_entities, cate, attn, to_entities)
        if self.args.display == 1:
            print(te)
        cand_paths, _, _ = generate_candidate_path_via_sparql_pool(te, answers, to_entities)
        #print(cand_paths[-1])
        p, r, y, an, anlen, orig_l, l = obtain_candidate_path_features(to_entities2, cand_paths, [dic, dic2], te=te)

        _t = Variable(torch.LongTensor(t), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(t), requires_grad = False)
        _p = Variable(torch.LongTensor(p), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(p), requires_grad = False)
        _r = Variable(torch.LongTensor(r), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(r), requires_grad = False)
        _y = Variable(torch.LongTensor(y), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.LongTensor(y), requires_grad = False)
        _an = Variable(torch.FloatTensor(an), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(an), requires_grad = False)
        _anlen = Variable(torch.FloatTensor(anlen), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(anlen), requires_grad = False)
        _l = Variable(torch.FloatTensor(l), requires_grad = False).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(l), requires_grad = False)
        _final_logit, _final_prob = self.execution_engine(_t, _q, _p, _r, _y, _an, _anlen)
        reward_mask = np.max(orig_l, axis = 1).reshape((-1, 1))
        _loss = self.execution_engine.obtain_loss(_final_prob, _l, torch.FloatTensor((reward_mask >0).astype(float)).cuda())
        # print(_loss)
        # exit()

        final_prob = _final_prob.cpu().data.numpy() if self.args.use_gpu else _final_prob.data.numpy()
        final_prob = np.argmax(final_prob, axis = 1)
        #print(_attn[0, :, :])

        #_correlation_constraint = torch.sum(torch.bmm(_v_attn, _v_attn.transpose(2, 1)))#/torch.bmm(_sigma_attn, _sigma_attn.transpose(2, 1)))
        #print(torch.bmm(_v_attn, _v_attn.transpose(2, 1)))
        #print(correlation_constraint)
        #reward = torch.gather(_l, 1, torch.max(_final_prob, 1)[1].view(-1, 1)).view(-1) #raw_reward# * is_responses * limits
         #_correlation_constraint = _correlation_constraint.cuda() if self.args.use_gpu else _correlation_constraint
        #_max_constraint = _max_constraint.cuda() if self.args.use_gpu else _max_constraint
        #_constraint = _correlation_constraint# + _max_constraint

        if self.args.train_execution_engine == 1:
            self.ee_optimizer.zero_grad()
            _loss.backward()
            self.ee_optimizer.step()

        reward_output = orig_l[np.arange(final_prob.shape[0]), final_prob]
        if self.args.train_program_generator != 0:
            if self.args.train_program_generator == 1:
                raw_reward = orig_l[np.arange(final_prob.shape[0]), final_prob]
            elif self.args.train_program_generator == 2:
                raw_reward = np.max(orig_l, axis = 1)
            #print(raw_reward)
            reward = raw_reward#*np.array([len(t) for t in te])
            _reward = torch.tensor(reward).type(torch.FloatTensor)
            _reward = _reward.cuda() if self.args.use_gpu else _reward
            _reward = (_reward - _reward.mean()) / (_reward.std() + eps)

            self.tp_optimizer.zero_grad()
            self.topic_generator.reinforce_backward(_reward, _prob, _q, _atten)
            self.tp_optimizer.step()
        #print(self.topic_generator.emb_init.weight[10, :10])

        loss = _loss.cpu().data.numpy() if self.args.use_gpu else _loss.data.numpy()
        logits = _logits.cpu().data.numpy() if self.args.use_gpu else _logits.data.numpy()
        logits = softmax(logits)
        logits = np.argmax(logits, axis = 1)
        evals = orig_l[np.arange(final_prob.shape[0]), final_prob]
        return loss, evals, reward_output, r, orig_l

    def test(self, batch_data, coincident, dic):
        dic, dic2 = dic

        if self.args.model_type == 'TG':
            self.topic_generator.eval()
            t, q, els, elo, cate, alia, orig_l, el, _ = batch_data
            with torch.no_grad():
                if self.args.tp_model == 'CHARLSTM':
                    q, q_char = q
                    alia, alia_char = alia
                    _q_char = Variable(torch.LongTensor(q_char)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q_char))
                    _alia_char = Variable(torch.LongTensor(alia_char)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia_char))
                _els = Variable(torch.FloatTensor(els)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(els))
                _elo = Variable(torch.FloatTensor(elo)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(elo))
                _cate = Variable(torch.FloatTensor(cate)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(cate))
                _q = Variable(torch.LongTensor(q)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q))
                _alia = Variable(torch.LongTensor(alia)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia))
            if self.args.tp_model == 'CHARLSTM':
                _logits, _ = self.topic_generator(_q, _els, _elo, _cate, _alia, q_char=_q_char, alia_char = _alia_char)
            else:
                _logits, _ = self.topic_generator(_q, _els, _elo, _cate, _alia)

            logits = _logits.cpu().data.numpy() if self.args.use_gpu else _logits.data.numpy()
            prob = sigmoid(logits) > 0.5
            evals = np.sum(orig_l * prob, 1)/np.maximum(np.sum(orig_l, 1), 1e-10)
            #prob = softmax(logits)
            prob = np.argmax(logits, axis = 1)
            #evals = orig_l[np.arange(prob.shape[0]), prob]
            return evals, prob, els+elo, orig_l, el

        elif self.args.model_type == 'EE':
            self.execution_engine.eval()
            t, q, p, r, y, an, anlen, orig_l, l, elmos = batch_data
            with torch.no_grad():
                _t = Variable(torch.LongTensor(t)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(t))
                _q = Variable(torch.LongTensor(q)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q))
                _p = Variable(torch.LongTensor(p)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(p))
                _r = Variable(torch.LongTensor(r)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(r))
                _y = Variable(torch.LongTensor(y)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(y))
                _an = Variable(torch.FloatTensor(an)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(an))
                _anlen = Variable(torch.FloatTensor(anlen)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(anlen))

            _logits, _prob = self.execution_engine(_t, _q, _p, _r, _y, _an, _anlen, elmos = elmos)

            probs = _prob.cpu().data.numpy() if self.args.use_gpu else _prob.data.numpy()
            prob = np.argmax(probs, axis = 1)
            probk = probs.argsort(axis = 1)[:, -10:]
            evals = orig_l[np.arange(prob.shape[0]), prob]
            return evals, (probs, prob, probk, [0]), r, orig_l, prob

        elif self.args.model_type == 'TG+EE':
            self.topic_generator.eval()
            self.execution_engine.eval()
            t, q, els, elo, cate, alia, orig_l, el, others = batch_data
            topic_entities, answers, to_entities, to_entities2 = others
            #print(topic_entities)
            with torch.no_grad():
                if self.args.tp_model == 'CHARLSTM':
                    q, q_char = q
                    alia, alia_char = alia
                    _q_char = Variable(torch.LongTensor(q_char)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q_char))
                    _alia_char = Variable(torch.LongTensor(alia_char)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia_char))
                _els = Variable(torch.FloatTensor(els)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(els))
                _elo = Variable(torch.FloatTensor(elo)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(elo))
                _cate = Variable(torch.FloatTensor(cate)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(cate))
                _q = Variable(torch.LongTensor(q)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(q))
                _alia = Variable(torch.LongTensor(alia)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(alia))

            if self.args.tp_model == 'CHARLSTM':
                _logits, _ = self.topic_generator(_q, _els, _elo, _cate, _alia, q_char = _q_char, alia_char = _alia_char)
            else:
                _logits, _ = self.topic_generator(_q, _els, _elo, _cate, _alia)

            logits = _logits.cpu().data.numpy() if self.args.use_gpu else _logits.data.numpy()
            topic_evals = orig_l[np.arange(logits.shape[0]), np.argmax(logits, axis = 1)]
            topk, topkscore, _ = self.topic_generator.sample(_logits, topic_entities, alia, k = 5) # _logits
            te = get_topic_entity_via_topk(topk, topic_entities, cate, topkscore, to_entities)
            if self.args.display == 1:
                print(te)
            cand_paths, _, _ = generate_candidate_path_via_sparql_pool(te, answers, to_entities)
            p, r, y, an, anlen, orig_l, l = obtain_candidate_path_features(to_entities2, cand_paths, [dic, dic2], te=te)
            #print(cand_paths)
            max_l = np.max(orig_l, axis = 1)
            with torch.no_grad():
                _t = Variable(torch.LongTensor(t)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(t))
                _p = Variable(torch.LongTensor(p)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(p))
                _r = Variable(torch.LongTensor(r)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(r))
                _y = Variable(torch.LongTensor(y)).cuda() if self.args.use_gpu else Variable(torch.LongTensor(y))
                _an = Variable(torch.FloatTensor(an)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(an))
                _anlen = Variable(torch.FloatTensor(anlen)).cuda() if self.args.use_gpu else Variable(torch.FloatTensor(anlen))

            _logits, _prob = self.execution_engine(_t, _q, _p, _r, _y, _an, _anlen)
            probs = _prob.cpu().data.numpy() if self.args.use_gpu else _prob.data.numpy()
            prob = np.argmax(probs, axis = 1)
            probk = probs.argsort(axis = 1)[:, -10:]
            evals = orig_l[np.arange(prob.shape[0]), prob]
            return evals, (probs, prob, probk, topic_evals), r, orig_l, (te, cand_paths)

class Train_loop(object):

    def __init__(self, args, emb):
        if args.use_gpu:
            torch.cuda.set_device(args.use_gpu)
        self.args = args
        self.emb = emb
        self.model = Model(args = args, emb = emb).cuda() if args.use_gpu else Model(args = args, emb = emb)
        self.save_para = self.model.state_dict().keys()
        if args.use_elmo:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device = args.use_gpu)
        else:
            self.elmo = False
        del emb

    def train(self, train, dev, test, dic, outfile):
        args = self.args
        reward_moving_average = 0, 0, 0

        dic2 = {}
        for d in dic:
            dic2[dic[d]] = d

        min_dev_eval, min_test_eval = -1, -1

        if args.load_id:
            # variables = self.model.execution_engine.state_dict()
            # for v in variables:
            #     print('%s\t%s' %(v, variables[v].size()))
            # variables = torch.load('%s/model%s.ckpt' %(outfile, args.load_id))
            # for v in variables:
            #     if 'classifier' in v:
            #         print('%s\t%s' %(v, variables[v]))
            # exit()
            try:
                if True:
                    self.model.load_state_dict(torch.load('%s/model%s.ckpt' %(outfile, args.load_id)), strict=False)
                else:
                    self.model.load_state_dict(torch.load('%s/modelTG+EE2_copy.ckpt' %(outfile)), strict=False)
                    #self.model.load_state_dict(torch.load('%s/modelEE2.ckpt' %(outfile)), strict=False)
                print('successfully load pre-trained parameters ...')
            except:
                print('fail to load pre-trained parameters ...')

        for ep in range(args.max_epoches):
            processed = 0
            train_cost = 0.
            train_loss = 0.
            train_pred = []
            train_eval = 0.
            train_bo_len = []

            if args.is_train:
                train_shuffle_batch = create_batches(train.len(), args.batch, is_shuffle = True)#[:4000]
            N = len(train_shuffle_batch) if args.is_train else 0

            for i in range(N):
                #print(train_shuffle_batch[i])
                if args.model_type == 'EE':
                    t, q, p, r, y, an, anlen, orig_l, l, elmos = obtain_topic_entity_candidate_path_features(
                                                    train, train_shuffle_batch[i], [dic, dic2], is_EE = True, use_elmo = self.elmo)
                    print('>>> batch: %s task %s saveid %s q %s r %s t %s p %s' %(i, args.task,
                        args.save_id, str(q.shape), str(r.shape), str(t.shape), str(p.shape)))
                else:
                    t, q, els, elo, cate, alia, orig_l, el, others = obtain_topic_entity_candidate_path_features(
                                                    train, train_shuffle_batch[i], [dic, dic2], is_CHARLSTM = (self.args.tp_model == 'CHARLSTM'))
                    q_ = q[0] if self.args.tp_model == 'CHARLSTM' else q
                    print('>>> batch: %s task %s saveid %s eln %s els %s t %s p %s' %(i, args.task,
                        args.save_id, str(els.shape), str(elo.shape), str(t.shape), str(q_.shape)))

                if args.model_type == 'TG':
                    # Train topic entity generator with ground-truth programs
                    cost, evals, pred, an, orig_l = self.model.TG_forward(q, els, elo, cate, alia, orig_l, el, [dic, dic2])
                elif args.model_type == 'EE':
                    # Train execution engine with ground-truth programs
                    cost, evals, pred, an = self.model.EE_forward(t, q, p, r, y, an, anlen, orig_l, l, elmos, [dic, dic2])
                elif args.model_type == 'TG+EE':
                    if i % 100 == 0:
                        self.model.args.train_execution_engine = 1
                    cost, evals, pred, an, orig_l = self.model.TGEE_forward(t, els, elo, cate, q, alia, orig_l, el, others,
                        train.coincident, [dic, dic2])

                #exit()
                k = len(pred)
                processed += k
                train_cost += cost
                train_loss += np.sum(pred)
                train_eval += np.sum(evals)
                train_pred += list(pred)
                #exit()

                print('loss\t%s' %(cost/k))
                for b in range(k):
                    cand_path_num = np.sum(an[b]> 0) if args.model_type in ['TG'] else np.sum(np.max(an[b], 1)> 0)
                    if i % args.display == 0:
                        print('index %s cand_path_num %s eval %s target %s' %(train_shuffle_batch[i][b], cand_path_num, evals[b], np.max(orig_l[b])))

                train_bo_len += [np.sum(an[b]>0)]

                if (i+1) % 100000 == 0:
                    embedding1 = self.model.topic_generator.emb_init.weight.cpu().data.numpy() if self.args.use_gpu else self.model.topic_generator.emb_init.weight.data.numpy()
                    embedding2 = self.model.execution_engine.emb_init.weight.cpu().data.numpy() if self.args.use_gpu else self.model.execution_engine.emb_init.weight.data.numpy()
                    np.save('%s/weights%s' %(outfile, args.save_id), [embedding1, embedding2])
                    saver = get_weights_and_biases(self.model, self.save_para)
                    torch.save(saver, '%s/model%s.ckpt' %(outfile, args.save_id))

            train_cost /= np.max([processed, 1.e-10])
            train_loss /= np.max([processed, 1.e-10])
            train_eval /= np.max([processed, 1.e-10])

            message = '%d | train loss: %g (%g) | train eval: %g' %(ep+1, train_cost, train_loss, train_eval)

            print('Evaluation ...')

            if args.only_eval:
                shuffle_batch = [[7]]
            else:
                shuffle_batch = create_batches(dev.len(), args.batch)

            N = len(shuffle_batch)
            processed = 0
            dev_eval = 0
            dev_pred = []

            for i in range(len(shuffle_batch)):
                if args.model_type == 'EE':
                    batch_data = obtain_topic_entity_candidate_path_features(
                                                        dev, shuffle_batch[i], [dic, dic2], is_EE = True, use_elmo = self.elmo)
                else:
                    batch_data = obtain_topic_entity_candidate_path_features(
                                                        dev, shuffle_batch[i], [dic, dic2], use_elmo = self.elmo, is_CHARLSTM = (self.args.tp_model == 'CHARLSTM'))

                evals, pred, an, orig_l, display = self.model.test(batch_data, dev.coincident, [dic, dic2])

                k = len(pred)
                processed += k
                dev_eval += np.sum(evals)
                dev_pred += list(pred)

                for b in range(k):
                    if args.only_eval:
                        print('\n>> Q: %s' %(index2sents(batch_data[1][b, :], dic2)))
                        if args.model_type == 'TG+EE':
                            print(relations[0])
                            relations = (display[0][b], sorted(display[1][b]))
                            if pred[b] < len(relations[1]):
                                print(','.join(relations[0]) + ' | ' + relations[1][pred[b]])
                            else:
                                print(','.join(relations[0]))
                    cand_path_num = np.sum(an[b]> 0) if args.model_type in ['TG'] else np.sum(np.max(an[b], 1)> 0)
                    if i % args.display == 0:
                        print('index %s cand_path_num %s eval %s target %s' %(shuffle_batch[i][b], cand_path_num, evals[b], np.max(orig_l[b])))

            dev_eval /= np.max([processed, 1.e-10])
            message += ' |val eval: %g' %dev_eval

            test_eval = 0.
            if not args.only_eval:
                if dev_eval > -np.inf:
                    shuffle_batch = create_batches(test.len(), args.batch)

                    processed = 0
                    test_eval = 0
                    test_maxeval = 0
                    test_pred = []
                    test_pred_output = []
                    test_te = []
                    topic_eval = 0
                    N = len(shuffle_batch)

                    for i in range(N):
                        if args.model_type == 'EE':
                            time1 = time.time()
                            batch_data = obtain_topic_entity_candidate_path_features(
                                                                test, shuffle_batch[i], [dic, dic2], is_EE = True, use_elmo = self.elmo)
                        else:
                            batch_data = obtain_topic_entity_candidate_path_features(
                                                                test, shuffle_batch[i], [dic, dic2], use_elmo = self.elmo, is_CHARLSTM = (self.args.tp_model == 'CHARLSTM'))
                        evals, pred, an, orig_l, display = self.model.test(batch_data, test.coincident, [dic, dic2])
                        if args.model_type in ["TG+EE", "EE"]:
                            probs, pred, predk, topic_evals = pred
                            probs = softmax(probs)

                        k = len(pred)
                        processed += k
                        test_eval += np.sum(evals)
                        test_maxeval += np.sum(np.max(orig_l, axis = 1))
                        test_pred += list(pred)
                        if args.model_type in ["TG+EE", "EE"]:
                            topic_eval += np.sum(topic_evals)
                            #topic_eval += np.sum([len(x) for x in display[0]])

                        for b in range(k):
                            if args.model_type == 'TG':
                                relations = batch_data[-1][0][b]
                            elif args.model_type == 'EE':
                                relations = test[shuffle_batch[i][b]][3]
                                #relations = test[shuffle_batch[i][b]][5]
                            else:
                                mid, relations = (display[0][b], display[1][b])
                            if len(relations) > 0 and (pred[b] < len(relations)):
                                if args.model_type in ['EE', 'TG+EE']:
                                    if probs is not None:
                                        if args.model_type == 'TG+EE':
                                            test_pred_output_unit = ','.join(mid) + ' | '
                                        else:
                                            test_pred_output_unit = ''
                                        for j in range(np.min([10, predk.shape[1]])):
                                            if predk[b, -(j+1)] < len(list(relations)):
                                                test_pred_output_unit += '\t' + list(relations)[predk[b, -(j+1)]] + '|' + str(probs[b, predk[b, -(j+1)]])
                                    test_pred_output += [test_pred_output_unit]
                                else:
                                    test_pred_output += [relations[pred[b]]]
                                    # print(relations[pred[b]])
                                    # print(relations[np.argmax(orig_l[b])])
                                    # print(display[b][:10])
                                    # print('%s\t%s' %(pred[b], np.argmax(orig_l[b])))
                            else:
                                test_pred_output += ['']
                            cand_path_num = np.sum(an[b]> 0) if args.model_type in ['TG'] else np.sum(np.max(an[b], 1)> 0)
                            if i % args.display == 0:
                                print('index %s cand_path_num %s eval %s target %s' %(shuffle_batch[i][b], cand_path_num, evals[b], np.max(orig_l[b])))

                    test_eval /= np.max([processed, 1.e-10])
                    test_maxeval /= np.max([processed, 1.e-10])
                    print('upper bound %s' %test_maxeval)
                    if args.model_type in ["TG+EE", "EE"]:
                        print('entity linking accuracy %s ' %(topic_eval*1./np.max([processed, 1.e-10])))
                    message += ' | test eval: %g' %test_eval

                    criteria = 0 if (args.model_type in ['TG+EE'] and ('WBQ' in args.test_q)) else min_test_eval
                    if test_eval >= criteria: #args.is_train and min_test_eval
                        test_preds = print_pred(test_pred_output, shuffle_batch)
                        save_pred('%s/pred%s_test10.txt' %(outfile, args.save_id), test_preds)
                        if args.model_type in ['TG+EE'] and ('WBQ' in args.test_q):
                            final_eval = GenerateFinalPredictions.main(args.save_id)
                            if final_eval > min_test_eval:
                                save_pred('%s/pred%s_test10_final_eval.txt' %(outfile, args.save_id), test_preds)
                                min_test_eval = final_eval
                                test_eval = final_eval
                            message += ' %s ' %final_eval
                        else:
                            min_test_eval = test_eval

            if args.is_train:
                if (test_eval == 0 and dev_eval > min_dev_eval) or (test_eval > 0 and test_eval == min_test_eval):
                    min_dev_eval = dev_eval
                    if args.model_type in ['TG', 'TG+EE']:
                        embedding1 = self.model.topic_generator.emb_init.weight.cpu().data.numpy() if self.args.use_gpu else self.model.topic_generator.emb_init.weight.data.numpy()
                    if args.model_type in ['TG+EE', 'EE']:
                        embedding2 = self.model.execution_engine.emb_init.weight.cpu().data.numpy() if self.args.use_gpu else self.model.execution_engine.emb_init.weight.data.numpy()
                    if args.model_type in ['TG']:
                        np.save('%s/weights%s' %(outfile, args.save_id), [embedding1, embedding1])
                    elif args.model_type in ['EE']:
                        np.save('%s/weights%s' %(outfile, args.save_id), [embedding2, embedding2])
                    elif args.model_type in ['TG+EE']:
                        np.save('%s/weights%s' %(outfile, args.save_id), [embedding1, embedding2])
                    saver = get_weights_and_biases(self.model, self.save_para)
                    torch.save(saver, '%s/model%s.ckpt' %(outfile, args.save_id))
                    message += '(saved model)'
            else:
                print(message)
                exit()

            log = open('%s/result%s.txt' %(outfile, args.save_id), 'a')
            log.write(message + '\n')
            log.close()

            print(message)
