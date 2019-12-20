import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):

	def __init__(self, method, hidden_size):
		super(Attention, self).__init__()
		self.method = method
		self.hidden_size = hidden_size

		if self.method == 'general':
			self.attention = nn.Linear(self.hidden_size, self.hidden_size)
		elif self.method == 'concat':
			self.attention = nn.Linear(self.hidden_size*2, self.hidden_size)
			self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

	def forward(self, hidden, encoder_outputs):
		seq_len = len(encoder_outputs)
		energies = Variable(torch.zeros(seq_len))#.cuda()
		for i in range(seq_len):
			energies[i] = self._score(hidden, encoder_outputs[i])
		return F.softmax(energies).unsqueeze(0).unsqueeze(0)

	def _score(self, hidden, encoder_output):
		if self.method == 'dot':
			energy = hidden.dot(encoder_output)
		elif self.method == 'general':
			energy = self.attention(encoder_output)
			energy = hidden.view(-1).dot(energy.view(-1))
		elif self.method == 'concat':
			energy = self.attention(torch.cat((hidden, encoder_output)))
			energy = self.other.dor(energy)
		return energy