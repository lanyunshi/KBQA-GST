from tqdm import tqdm
import random
import numpy as np

import torch
from torch.autograd import Variable

from . import utils

def idx2word(indexes, vocab2):
	return [vocab2[x] for x in indexes]

class Trainer(object):
	def __init__(self, args, encoder, decoder, encoder_opt, decoder_opt, criterion):
		super(Trainer, self).__init__()
		self.args = args
		self.encoder = encoder
		self.decoder = decoder
		self.criterion = criterion
		self.encoder_opt = encoder_opt
		self.decoder_opt = decoder_opt
		self.epoch = 0

	def train(self, dataset):
		#self.model.train()
		self.encoder_opt.zero_grad()
		self.decoder_opt.zero_grad()
		total_loss = 0.0

		indices = torch.randperm(len(dataset), dtype=torch.long)
		for idx in tqdm(range(len(dataset)), 
			desc='Train epoch ' + str(self.epoch + 1) + ''):
			nlq, arg, nlqtree = dataset[indices[idx]]
			word_candidate = [2, 3] + list(np.unique(nlq.numpy(), return_inverse=True)[0])
			loss = 0.0

			target = torch.from_numpy(np.array([word_candidate.index(x) for x in arg.numpy()])).view(-1, 1)
			target_length = arg.size()[0]

			encoder_hidden = self.encoder.init_hidden()
			encoder_outputs, encoder_hidden = self.encoder(nlq, encoder_hidden)

			decoder_input = Variable(torch.LongTensor([0]))
			#decoder_input = decoder_input.cuda()
			decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
			#decoder_context = decoder_context.cuda()
			decoder_hidden = encoder_hidden

			use_teacher_forcing = random.random() < self.args.use_teacher_forcing_ratio
			if use_teacher_forcing:
				for di in range(target_length):
					decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
									 decoder_input,
									 torch.from_numpy(np.array(word_candidate)),
									 decoder_context,
									 decoder_hidden,
									 encoder_outputs)
					loss += self.criterion(decoder_output, target[di])
					decoder_input = target[di]
			else:
				for di in range(target_length):
					decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
									 decoder_input,
									 torch.from_numpy(np.array(word_candidate)),
									 decoder_context,
									 decoder_hidden,
									 encoder_outputs)
					loss += self.criterion(decoder_output, target[di])

					topv, topi = decoder_output.data.topk(1)
					ni = topi[0][0]

					decoder_input = Variable(torch.LongTensor([[ni]]))
					#decoder_input = decoder_input.cuda()

					if word_candidate[int(ni.numpy())] == 1:
						break

			total_loss += loss.data[0]
			loss.backward()
			torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.args.clip)
			torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.args.clip)
			self.encoder_opt.step()
			self.decoder_opt.step()

		self.epoch += 1
		return total_loss/len(dataset)

	def test(self, dataset):
		#self.model.eval()
		with torch.no_grad():
			total_loss = 0.0

		pred_args = []
		for idx in tqdm(range(len(dataset))):
			nlq, arg, nlqtree = dataset[idx]
			word_candidate = [2, 3] + list(np.unique(nlq.numpy(), return_inverse=True)[0])
			loss = 0.0

			target = torch.from_numpy(np.array([word_candidate.index(x) for x in arg.numpy()])).view(-1, 1)
			target_length = arg.size()[0]
			#print(idx2word(arg.numpy(), self.args.vocab2))
			#print(target_length)

			encoder_hidden = self.encoder.init_hidden()
			encoder_outputs, encoder_hidden = self.encoder(nlq, encoder_hidden)

			decoder_input = Variable(torch.LongTensor([0]))
			#decoder_input = decoder_input.cuda()
			decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
			#decoder_context = decoder_context.cuda()
			decoder_hidden = encoder_hidden

			pred_arg = []
			for di in range(target_length):
				decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
								decoder_input,
								torch.from_numpy(np.array(word_candidate)),
								decoder_context,
								decoder_hidden,
								encoder_outputs)
				loss += self.criterion(decoder_output, target[di])

				#print(decoder_output.data)
				topv, topi = decoder_output.data.topk(1)
				ni = topi[0][0]

				decoder_input = Variable(torch.LongTensor([[ni]]))
				#decoder_input = decoder_input.cuda()
				pred_arg += [word_candidate[int(ni.numpy())]]

				if word_candidate[int(ni.numpy())] == 1:
					break
			#print(pred_arg)

			total_loss += loss.data[0]
#			print(pred_arg)
			pred_args += [pred_arg]
		return total_loss/len(dataset), pred_args