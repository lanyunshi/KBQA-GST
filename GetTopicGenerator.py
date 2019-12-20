import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import numpy as np
from torch.distributions import Categorical

class ConvNet(nn.Module):
    def __init__(self, max_length, Cin, Cout, kernel_size = 5, stride = 1, padding = 2, dropout = 0.2):
        super(ConvNet, self).__init__()
        self.max_length = max_length
        self.layer = nn.Sequential(
            nn.Conv1d(Cin, Cout, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm1d(Cout),
            nn.Dropout(dropout),
            nn.ReLU()
        )
    def forward(self, x):
        x = x.transpose(2, 1)
        out = self.layer(x)
        out, _ = torch.max(out, dim = 2)
        return out

class LSTMNet(nn.Module):
    def __init__(self, Cin, Cout, use_gpu = False, dropout = 0.2, max_pooling = True):
        super(LSTMNet, self).__init__()
        self.n_h = Cout
        self.max_pooling = max_pooling
        self.use_gpu = use_gpu
        self.lstm = nn.LSTM(Cin, Cout, 1, bidirectional=True, batch_first = True)
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.n_h*2, self.n_h),
            #nn.BatchNorm1d(Cout),
            #nn.ReLU()
        )
    def forward(self, x):
        x = x.transpose(2, 1)
        B, l, n_e = x.size()
        h0 = torch.zeros((2, B, self.n_h)).cuda() if self.use_gpu else torch.zeros((2, B, self.n_h))
        c0 = torch.zeros((2, B, self.n_h)).cuda() if self.use_gpu else torch.zeros((2, B, self.n_h))
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer(out)
        if self.max_pooling:
            out, _ = torch.max(out, dim = 1)
        return out

class TopicGenerator(nn.Module):
    def __init__(self, args, emb):
        super(TopicGenerator, self).__init__()
        self.args = args
        self.emb = emb
        self.vocab_size, self.emb_dim = emb.shape
        self.hidden_dim = args.hidden_dim

        self.emb_init = nn.Embedding(self.emb.shape[0], self.emb.shape[1])
        self.emb_init.weight.data.copy_(torch.from_numpy(self.emb))
        self.q_linear = nn.Linear(self.hidden_dim, self.emb_dim)
        self.classifier = nn.Linear(5, 1)
        if args.tp_model == 'CNN':
            self.q_encoder = ConvNet(args.max_plen, self.emb_dim, self.hidden_dim, args.kernel_size, args.stride, args.padding)
        elif args.tp_model == 'LSTM':
            self.q_encoder = LSTMNet(self.emb_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout)
            self.q_linear = nn.Linear(self.emb_dim, self.hidden_dim)
            self.bmm = nn.Linear(2*self.hidden_dim, 1)
        elif self.args.tp_model == 'CoAttention':
            self.q_encoder = nn.Sequential(nn.Linear(2*self.emb_dim, self.hidden_dim), nn.ReLU())#LSTMNet(int(2*self.emb_dim), int(self.hidden_dim), use_gpu = args.use_gpu, dropout = args.dropout)
            self.similar = nn.Linear(int(3*self.emb_dim), 1)
            self.q_linear = nn.Linear(self.hidden_dim, 1)
        elif args.tp_model == 'CHARLSTM':
            self.q_encoder = LSTMNet(self.hidden_dim, self.hidden_dim)
            self.char_encoder = LSTMNet(self.hidden_dim, self.hidden_dim)
            self.q_linear = nn.Linear(self.emb_dim, self.hidden_dim)
            self.classifier = nn.Linear(6, 1)
        elif args.tp_model == 'Sum':
            self.q_encoder = LSTMNet(int(self.emb_dim), int(self.emb_dim), use_gpu = args.use_gpu, dropout = args.dropout, max_pooling = False)
        # self attention
        self.softmax = torch.nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        # attention of q
        #self.softmax = torch.nn.Softmax(dim = 3)
        # basic and self attention
        # attention of q
        #self.classifier = nn.Linear(self.hidden_dim + 4, 1)
        self.criterion = nn.KLDivLoss()

        self.lr = args.learning_rate
        self.dropout = args.dropout

    def forward(self, q, els, elo, cate, alia, q_char=None, alia_char=None):
        padding_id = 0
        n_h = self.hidden_dim
        n_e = self.emb_dim
        B, ql = q.size()
        B, TN, al = alia.size()
        if self.args.tp_model == 'CHARLSTM':
            B, ql, qwl = q_char.size()
            B, TN, al, awl = alia_char.size()

        self.emb_init.weight.data = F.normalize(self.emb_init.weight.data, p =2, dim = 1)
        self.emb_init.weight.data[0, :].fill_(0)

        els = els.view(B, TN, 1)
        elo = elo.view(B, TN, 1)

        q_emb = self.emb_init(q).view(B, ql, n_e)
        alia_emb = self.emb_init(alia).view(B, TN, al, n_e)
        if self.args.tp_model == 'CHARLSTM':
            q_char_emb = self.emb_init(q_char).view(B*ql, qwl, n_e)
            alia_char_emb = self.emb_init(alia_char).view(B*TN*al, awl, n_e)

        # matching at character level
        if self.args.tp_model == 'CHARLSTM':
            q_emb = self.q_linear(q_emb)
            q_rep = self.q_encoder(q_emb, self.args.use_gpu).view(B, 1, n_h)
            alia_emb = self.q_linear(alia_emb)
            alia_rep = torch.mean(alia_emb, dim = 2)
            alia_score = torch.bmm(alia_rep, q_rep.transpose(2, 1)).view(B, TN, 1)

            q_char_emb = self.q_linear(q_char_emb)
            alia_char_emb = self.q_linear(alia_char_emb)
            q_char_rep = self.char_encoder(q_char_emb, self.args.use_gpu).view(B, ql, n_h)
            alia_char_rep = self.char_encoder(alia_char_emb, self.args.use_gpu).view(B, TN, al, n_h)
            q_char_rep = self.q_encoder(q_char_rep, self.args.use_gpu).view(B, 1, n_h)
            alia_char_rep = torch.mean(alia_emb, dim = 2)
            alia_char_score = torch.bmm(alia_char_rep, q_char_rep.transpose(2, 1)).view(B, TN, 1)

            input = torch.cat((els, elo, cate, alia_score, alia_char_score), dim = 2).view(B, TN, 6)

        elif self.args.tp_model == 'CoAttention':
            q_mask = 1 - torch.eq(q, padding_id).type(torch.FloatTensor).view(B, ql, 1)
            alia_mask = 1 - torch.eq(alia, padding_id).type(torch.FloatTensor).view(B, TN*al, 1)
            mask = torch.bmm(alia_mask, q_mask.transpose(2, 1)).cuda()
            #atten = torch.bmm(alia_emb.view(B, TN*al, n_e), q_emb.transpose(2, 1))
            w = alia_emb.view(B, TN*al, 1, n_e).repeat(1, 1, ql, 1)
            v = q_emb.view(B, 1, ql, n_e).repeat(1, TN*al, 1, 1)
            atten = self.similar(torch.cat([w, v, w*v], -1)).view(B, TN*al, ql)
            mask_values = -1e10*torch.ones_like(mask).cuda()
            atten = (mask*atten + (1-mask)*mask_values).view(B, TN, al, ql)
            atten1 = F.softmax(atten, 3).view(B, TN*al, ql)
            alig_a = torch.bmm(atten1, q_emb).view(B*TN, al, n_e)
            compare = torch.sum(self.q_encoder(torch.cat([alig_a, alia_emb.view(B*TN, al, n_e)], -1)).view(B, TN, al, -1), 2)
            alia_score = self.q_linear(compare).view(B, TN, 1)

            input = torch.cat((elo, els, cate, alia_score), dim = 2).view(B, TN, 5) # elo,

        else:
            if self.args.tp_model in ['CNN', 'LSTM']:
                # directly use last hidden representation of q, keep this!
                q_rep = self.q_encoder(q_emb.transpose(2, 1)).view(B, 1, n_h)

                # self attention of q
                q_score = self.bmm(torch.cat((self.q_linear(q_emb), q_rep.repeat(1, ql, 1)), dim = 2)).view(B, ql)
                mask = (1 - torch.eq(q, padding_id).type(torch.FloatTensor).view(B, ql)).cuda()
                mask_values = (-1.e10*torch.ones_like(mask)).cuda()
                q_score = mask*q_score + (1-mask)*mask_values
                q_score = self.softmax(q_score).view(B, ql, 1)
                q_rep = torch.sum(q_score*q_emb, dim = 1).view(B, 1, n_e)
            elif self.args.tp_model in ['Sum']:
                q_rep = self.q_encoder(q_emb.transpose(2, 1)).view(B, ql, n_e)
                q_rep = torch.sum(q_rep, dim = 1).view(B, 1, n_e)

            q_mask = 1 - torch.eq(q, padding_id).type(torch.FloatTensor).view(B, ql, 1)
            alia_mask = 1 - torch.eq(alia, padding_id).type(torch.FloatTensor).view(B, TN*al, 1)
            mask = torch.bmm(alia_mask, q_mask.transpose(2, 1)).cuda()
            #atten = torch.bmm(alia_emb.view(B, TN*al, n_e), q_emb.transpose(2, 1))
            atten = self.similar(torch.cat([alia_emb.view(B, TN*al, 1, n_e).repeat(1, 1, ql, 1), q_emb.view(B, 1, ql, n_e).repeat(1, TN*al, 1, 1)], -1)).view(B, TN*al, ql)
            mask_values = -1e10*torch.ones_like(mask).cuda()
            atten = (mask*atten + (1-mask)*mask_values).view(B, TN, al, ql)
            atten1 = F.softmax(atten, 3).view(B, TN*al, ql)
            alia_rep = torch.mean(alia_emb, dim = 2)
            atten1 = torch.sum(atten1.view(B, TN, al, ql), dim = 2)

            # attention of q
            # q_mask = 1 - torch.eq(q, padding_id).type(torch.FloatTensor).view(B, ql, 1)
            # alia_mask = 1 - torch.eq(alia, padding_id).type(torch.FloatTensor).view(B, TN*al, 1)
            # mask = torch.bmm(alia_mask, q_mask.transpose(2, 1)).cuda()
            # atten = torch.bmm(alia_emb.view(B, TN*al, n_e), q_emb.transpose(2, 1))
            # mask_values = -1e10*torch.ones_like(mask).cuda()
            # atten = (mask*atten + (1-mask)*mask_values).view(B, TN, al, ql)
            # atten = F.softmax(atten).view(B, TN*al, ql)
            # alig_alia = torch.bmm(atten, q_emb).view(B, TN, al, n_e)
            # diff_alia = ((alia_emb - alig_alia)**2).view(B*TN, al, n_e)
            # alia_score = self.q_encoder(diff_alia, self.args.use_gpu).view(B, TN, n_h)

            # transform, basic
            #alia_score = torch.bmm(alia_rep, self.q_linear(q_rep).transpose(2, 1)).view(B, TN, 1)

            # self attention, don't need transform
            alia_score = torch.bmm(alia_rep, q_rep.transpose(2, 1)).view(B, TN, 1)

            # basic and self attention
            input = torch.cat((elo, els, cate, alia_score), dim = 2).view(B, TN, 5) #els,

        # attention of q
        #input = torch.cat((els, elo, cate, alia_score), dim = 2).view(B, TN, n_h+4)

        logits = self.classifier(input).view(B, TN)
        mask = 1 - torch.eq(els + elo, padding_id).type(torch.FloatTensor).view(B, TN)
        mask = mask.cuda() if self.args.use_gpu else mask
        mask_values = -1.e10*torch.ones_like(mask)
        mask_values = mask_values.cuda() if self.args.use_gpu else mask_values
        mask_logits = mask*logits + (1-mask)*mask_values
        max_atten, _ = torch.max(atten.view(B, TN, al, ql), 2)
        _max_atten = Variable(max_atten, requires_grad = False).cuda()
        atten1 = F.softmax(torch.sum(F.softmax(mask_logits, 1).view(B, TN, 1)*_max_atten, 1), 1)

        return mask_logits, atten1

    def obtain_loss(self, p, l):
        return torch.sum(self.criterion(F.softmax(p, 1), l))
        #return F.binary_cross_entropy(F.sigmoid(p), l)

    def sample(self, p, topic_entities, alia, k = 3):
        B, TN = p.size()
        mask = np.zeros((B, TN), dtype=np.float32)
        for i in range(len(topic_entities)):
            mask[i, :len(topic_entities[i])] = 1
        max_topic_num = np.max([len(topic_entities[i]) for i in range(len(topic_entities))])
        mask = torch.FloatTensor(mask)
        mask = mask.cuda() if self.args.use_gpu else mask
        p = mask * p + (-1e10)* (1-mask)
        p = F.softmax(p, dim = 1)
        y = np.zeros((B, max_topic_num), dtype = np.int32)
        ys = np.zeros((B, max_topic_num), dtype = np.float32)
        for i in range(B):
            ms = p.data[i, :].cpu().numpy() if self.args.use_gpu else p.data[i, :].numpy()
            kidx = np.array(ms.argsort()[::-1][:len(topic_entities[i])])
            y[i, :len(kidx)]= kidx
            ys[i, :len(ms[kidx])] = ms[kidx]
        return y, ys, p

    def reinforce_sample(self, p, topic_entities, alia = None, k = 3, atten = None):
        B, TN = p.size()
        if atten is not None:
            B, TN, ql = atten.size()
            attns = torch.zeros((B, k, ql), dtype = torch.float32)
        mask = np.zeros((B, TN), dtype=np.float32)
        for i in range(len(topic_entities)):
            mask[i, :len(topic_entities[i])] = 1
        mask = torch.FloatTensor(mask)
        mask = mask.cuda() if self.args.use_gpu else mask
        p = mask * p + (-1e10)* (1-mask)
        p = torch.softmax(p, dim =1)
        #print(p[0, :10])
        y = torch.zeros((B, k), dtype=torch.int32)
        probs = torch.zeros((B, k))
        m = Categorical(p)
        ms = m.sample((k, ))
        y = torch.t(ms)
        probs = torch.t(m.log_prob(ms))
        if atten is not None:
            for i in range(B):
                attns[i, :] = torch.index_select(p[i], 0, ms[:, i])
        else:
            attns = None

        return y, probs, p

    def reinforce_backward(self, reward, prob, q, atten):
        policy_loss = []
        for t in range(prob.shape[1]):
            policy_loss.append((-prob[:, t] * reward))  #

        q_mask = 1 - torch.eq(q, 0).type(torch.FloatTensor)
        q_mask = q_mask/torch.sum(q_mask, 1).view(-1, 1)
        constraint = torch.sum(self.criterion(atten, q_mask.cuda()))
        policy_loss = torch.cat(policy_loss).sum() + 0.4*constraint
        policy_loss.backward()
