import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderGRU(nn.Module):
	def __init__(self, vocab_size, hidden_size, n_layers = 1):
		super(EncoderGRU, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.emb = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

	def forward(self, word_inputs, hidden):
		seq_len = len(word_inputs)
		embedded = self.emb(word_inputs).view(seq_len, 1, -1)
		output, hidden = self.gru(embedded, hidden)
		return output, hidden

	def init_hidden(self):
		hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
		#hidden = hidden.cuda()
		return hidden