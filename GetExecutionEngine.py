import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time

class ConvNet(nn.Module):
    def __init__(self, Cin, Cout, kernel_size = 5, stride = 1, padding = 2, dropout = 0.2):
        super(ConvNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(Cin, Cout, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm1d(Cout),
            #nn.Dropout(dropout),
            #nn.ReLU()
        )
    def forward(self, x):
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

def KL_divergence(p, q):
    return p * torch.log(torch.clamp(p/q, min = 1.e-10))

class Ranker(nn.Module):

    def __init__(self, args, emb):
        super(Ranker, self).__init__()
        self.args = args
        self.emb = emb
        self.vocab_size, self.emb_dim = emb.shape
        self.hidden_dim = args.hidden_dim

        self.ACTIVATION_DICT = {'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        self.emb_init = nn.Embedding(self.emb.shape[0], self.emb.shape[1])
        self.emb_init.weight.data.copy_(torch.from_numpy(self.emb))
        # self.emb_init.weight.requires_grad=False
        if args.model == 'CNN':
            self.pattern_cnn = ConvNet(self.emb_dim, self.hidden_dim, args.kernel_size, args.stride, args.padding, args.dropout)
            self.type_cnn = ConvNet(self.emb_dim, self.hidden_dim, args.kernel_size, args.stride, args.padding, args.dropout)
            self.question_cnn = ConvNet(self.emb_dim, self.hidden_dim, args.kernel_size, args.stride, args.padding, args.dropout)
            # self.encoder = LSTMNet(self.emb_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout, max_pooling=False)
            if args.use_elmo:
                self.elmo_pattern_cnn = ConvNet(3072, self.hidden_dim, args.kernel_size, args.stride, args.padding, args.dropout)
                self.elmo_question_cnn = ConvNet(3072, self.hidden_dim, args.kernel_size, args.stride, args.padding, args.dropout)
        elif args.model == 'LSTM':
            self.pattern_cnn = LSTMNet(self.emb_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout)
            self.question_cnn = LSTMNet(self.emb_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout)
        elif args.model == 'HRLSTM':
            self.pattern_cnn = LSTMNet(self.emb_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout, max_pooling = False)
            self.pattern_cnn2 = LSTMNet(self.hidden_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout, max_pooling = False)
            self.question_cnn = LSTMNet(self.emb_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout, max_pooling = False)#
            self.question_cnn2 = LSTMNet(self.hidden_dim, self.hidden_dim, use_gpu = args.use_gpu, dropout = args.dropout, max_pooling = False)
        elif args.model == 'Attention':
            self.pattern_cnn = LSTMNet(int(2*self.emb_dim), int(self.hidden_dim/2), use_gpu = args.use_gpu, dropout = args.dropout)
            self.question_cnn = LSTMNet(int(2*self.emb_dim), int(self.hidden_dim/2), use_gpu = args.use_gpu, dropout = args.dropout)
            self.projector =nn.Linear(self.hidden_dim, 1)
        if args.model != 'Attention':
            self.pattern_projector = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.question_projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(5, 1) if args.use_elmo else nn.Linear(4, 1)
        self.criterion = nn.KLDivLoss()

        self.lr = args.learning_rate
        self.dropout = args.dropout

    def forward(self, t, q, p, r, y, an, anlen, elmos = False):
        padding_id = 0
        n_h = self.hidden_dim
        n_e = self.emb_dim
        B, tl = t.size()
        B, CN, pl = p.size()
        B, ql = q.size()
        B, CN, rl = r.size()
        B, CN, yl = y.size()
        if elmos: elmo_t, elmo_q, elmo_p, elmo_r = elmos
        if self.args.use_gpu and self.args.use_elmo:
            elmo_t = elmo_t.cuda()
            elmo_p = elmo_p.cuda()
            elmo_q = elmo_q.cuda()
            elmo_r = elmo_r.cuda()

        # self.emb_init.weight.data = F.normalize(self.emb_init.weight.data, p = 2, dim = 1)
        # self.emb_init.weight.data[0, :].fill_(0)

        t_emb = self.emb_init(t).view(B, tl, n_e).transpose(2, 1)
        p_emb = self.emb_init(p).view(B*CN, pl, n_e).transpose(2, 1)
        q_emb = self.emb_init(q).view(B, ql, n_e).transpose(2, 1)
        r_emb = self.emb_init(r).view(B*CN, rl, n_e).transpose(2, 1)
        y_emb = self.emb_init(y).view(B*CN, yl, n_e).transpose(2, 1)
        if self.args.model == 'HRLSTM':
            q_rep1 = self.pattern_cnn(q_emb)
            q_rep2 = self.pattern_cnn2(q_rep1.transpose(2, 1))
            q_rep = (q_rep1 + q_rep2)
            q_rep = torch.max(q_rep, dim = 1)[0].view(B, 1, n_h)
            p_rep = self.pattern_cnn(p_emb)
            r_rep = self.pattern_cnn(r_emb)
            p_rep = torch.cat([p_rep, r_rep], dim = 1)
            p_rep = torch.max(p_rep, dim = 1)[0].view(B, CN, n_h)
        elif self.args.model in ['LSTM', 'CNN']:
            activation = self.ACTIVATION_DICT[self.args.activation]

            t_rep = self.pattern_cnn(t_emb).view(B, 1, n_h)
            p_rep = self.pattern_cnn(p_emb).view(B, CN, n_h)
            q_rep = self.question_cnn(q_emb).view(B, 1, n_h)
            r_rep = self.question_cnn(r_emb).view(B, CN, n_h)
            ty_rep = self.type_cnn(t_emb).view(B, 1, n_h)
            y_rep = self.type_cnn(y_emb).view(B, CN, n_h)
        if elmos:
            elmo_t_rep = self.elmo_pattern_cnn(elmo_t.view(B, tl, 3072).transpose(2, 1)).view(B, 1, n_h)
            elmo_p_rep = self.elmo_pattern_cnn(elmo_p.view(B*CN, pl, 3072).transpose(2, 1)).view(B, CN, n_h)
            elmo_q_rep = self.elmo_question_cnn(elmo_q.view(B, ql, 3072).transpose(2, 1)).view(B, 1, n_h)
            elmo_r_rep = self.elmo_question_cnn(elmo_r.view(B*CN, rl, 3072).transpose(2, 1)).view(B, CN, n_h)

        if self.args.model == 'Attention':
            #t_emb = self.emb_init(t).view(B, tl, n_e)
            #p_emb = self.emb_init(p).view(B, CN, pl, n_e)
            q_emb = self.emb_init(q).view(B, ql, n_e)
            r_emb = self.emb_init(r).view(B, CN, rl, n_e)
            # t_mask = 1 - torch.eq(t, padding_id).type(torch.FloatTensor).view(B, tl, 1)
            # p_mask = 1 - torch.eq(p, padding_id).type(torch.FloatTensor).view(B, CN*pl, 1)
            # mask = torch.bmm(p_mask, t_mask.transpose(2, 1)).cuda()
            # atten = torch.bmm(p_emb.view(B, CN*pl, n_e), t_emb.transpose(2, 1))
            # mask_values = -1.e10*torch.ones_like(mask).cuda()
            # atten = (mask*atten + (1-mask)*mask_values).view(B, CN, pl, tl)
            # atten1 = F.softmax(atten, 3).view(B, CN*pl, tl)
            # alig_p = torch.bmm(atten1, t_emb).view(B*CN, pl, n_e)
            # atten2 = F.softmax(atten, 2).view(B*CN, pl, tl)
            # alig_t = torch.bmm(atten2.transpose(2, 1), p_emb.view(B*CN, pl, n_e)).view(B*CN, tl, n_e)
            # compare_p = self.pattern_cnn(torch.cat([alig_p, p_emb.view(B*CN, pl, n_e)], -1).transpose(2, 1)).view(B, CN, -1)
            # compare_t = self.pattern_cnn(torch.cat([alig_t, t_emb.unsqueeze(1).repeat(1, CN, 1, 1).view(B*CN, tl, n_e)], -1).transpose(2, 1)).view(B, CN, -1)

            q_mask = 1 - torch.eq(q, padding_id).type(torch.FloatTensor).view(B, ql, 1)
            r_mask = 1 - torch.eq(r, padding_id).type(torch.FloatTensor).view(B, CN*rl, 1)
            mask = torch.bmm(r_mask, q_mask.transpose(2, 1)).cuda()
            atten = torch.bmm(r_emb.view(B, CN*rl, n_e), q_emb.transpose(2, 1))
            mask_values = -1.e10*torch.ones_like(mask).cuda()
            atten = (mask*atten + (1-mask)*mask_values).view(B, CN, rl, ql)
            atten1 = F.softmax(atten, 3).view(B, CN*rl, ql)
            alig_r = torch.bmm(atten1, q_emb).view(B*CN, rl, n_e)
            atten2 = F.softmax(atten, 2).view(B*CN, rl, ql)
            alig_q = torch.bmm(atten2.transpose(2, 1), r_emb.view(B*CN, rl, n_e)).view(B*CN, ql, n_e)
            compare_r = self.question_cnn(torch.cat([alig_r, r_emb.view(B*CN, rl, n_e)], -1).transpose(2, 1)).view(B, CN, -1)
            compare_q = self.question_cnn(torch.cat([alig_q, q_emb.unsqueeze(1).repeat(1, CN, 1, 1).view(B*CN, ql, n_e)], -1).transpose(2, 1)).view(B, CN, -1)

            #p_score = self.projector(torch.cat([compare_p, compare_t], -1))
            r_score = self.projector(torch.cat([compare_r, compare_q], -1))
        else:
            p_score = torch.bmm(p_rep, self.pattern_projector(t_rep).transpose(2, 1)).view(B, CN, 1)
            r_score = torch.bmm(r_rep, self.question_projector(q_rep).transpose(2, 1)).view(B, CN, 1)
            y_score = torch.bmm(y_rep, self.pattern_projector(ty_rep).transpose(2, 1)).view(B, CN, 1)
        if elmos:
            elmo_p_score = torch.bmm(elmo_p_rep, self.pattern_projector(elmo_t_rep).transpose(2, 1)).view(B, CN, 1)
            elmo_r_score = torch.bmm(elmo_r_rep, self.pattern_projector(elmo_q_rep).transpose(2, 1)).view(B, CN, 1)
        an = an.view(B, CN, 1)
        anlen = torch.log(anlen.view(B, CN, 1)+1)
        # score = torch.cat(((ty_rep - y_rep)**2,
        #                     (t_rep - p_rep)**2), dim = 2)

        if elmos:  # I comment this out !
            input = torch.cat((r_score, p_score, elmo_r_score, elmo_p_score, an), dim = 2).view(B, CN, 5)
        else:
            input = torch.cat((y_score, r_score, p_score, an), dim = 2).view(B, CN, 4) # y_score,
        # input = score.view(B, CN, 2*n_h)
        #print(input[0, :10, :])
        logits = self.classifier(input).view(B, CN) # !!! I comment out this
        #logits = p_score.view(B, CN)

        r_mask = 1 - torch.eq(an, padding_id).type(torch.FloatTensor).view(B, CN)
        r_mask = r_mask.cuda() if self.args.use_gpu else r_mask
        mask_values = -1.e10*torch.ones_like(r_mask)
        mask_logits = r_mask*logits + (1-r_mask)*mask_values
        probs = F.softmax(mask_logits, 1)
        # mask_values = 0*torch.ones_like(r_mask)
        # mask_probs = r_mask*probs + (1-r_mask)*mask_values

        return logits, probs

    def obtain_loss(self, p, l, mask = None):
        # print(p[0, :])
        # print(l[0, :])
        if mask is None:
            loss = torch.sum(self.criterion(p, l))#torch.sum((p-l)**2)#
        else:
            loss = torch.sum(self.criterion(p, l)*mask)
        return loss
