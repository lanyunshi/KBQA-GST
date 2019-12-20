import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention

class AttentionDecoderRNN(nn.Module):
	def __init__(self, vocab_size, attention_model, hidden_size, n_layers=1, dropout_p = .1):
		super(AttentionDecoderRNN, self).__init__()
		self.attention_model = attention_model
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.n_layers = n_layers
		self.dropout_p = dropout_p

		self.emb = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout_p)
		self.out = nn.Linear(hidden_size*2, hidden_size)

		if attention_model is not None:
			self.attention = Attention(attention_model, hidden_size)

	def forward(self, word_input, word_candidate, last_context, last_hidden, encoder_outputs):
		word_embedded = self.emb(word_input).view(1, 1, -1)
		rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
		rnn_output, hidden = self.gru(rnn_input, last_hidden)

		attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
		context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		output = self.out(torch.cat((rnn_output, context), 1))
		out_word_embedded = self.emb(word_candidate).view(-1, self.hidden_size)
		output = F.log_softmax(torch.mm(output, torch.t(out_word_embedded)))

		return output, context, hidden, attention_weights
