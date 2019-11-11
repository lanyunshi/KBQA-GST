import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from .tree import Tree

class Dataset(data.Dataset):
	def __init__(self, path, vocab):
		super(Dataset, self).__init__()
		self.vocab = vocab

		self.nlqs, self.args = self.read_sentences(os.path.join(path, 'nlq2arg.toks'))
		self.nlqtrees = self.read_trees(os.path.join(path, 'nlq.parents'))

		self.size = len(self.nlqs)

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		nlq = deepcopy(self.nlqs[index])
		arg = deepcopy(self.args[index])
		nlqtree = deepcopy(self.nlqtrees[index])
		return (nlq, arg, nlqtree)

	def read_sentences(self, filename):
		nlq2indices = []
		arg2indices = []
		with open(filename, 'r') as f:
			for line_idx, line in enumerate(f):
				line = line.rstrip('\n')
				nlq, arg = line.split('\t')

				nlq2indice = self.read_sentence(nlq)
				arg2indice = self.read_sentence(arg)

				nlq2indices += [nlq2indice]
				arg2indices += [arg2indice]
		return nlq2indices, arg2indices

	def read_sentence(self, line):
		vec = []
		for w in line.split():
			if w in self.vocab:
				vec += [self.vocab[w]]
			else:
				vec += [self.vocab['UNK']]
		return torch.tensor(vec, dtype = torch.long)

	def read_trees(self, filename):
		nlq2trees = []
		with open(filename, 'r') as f:
			for line_idx, line in enumerate(f):
				nlq = line.rstrip()

				nlq2tree = self.read_tree(nlq)

				nlq2trees += [nlq2tree]
		return nlq2trees

	def read_tree(self, line):
		parents = list(map(int, line.split()))
		trees = dict()
		root = None
		for i in range(1, len(parents) + 1):
			if i - 1 not in trees.keys() and parents[i - 1] != -1:
				#print('i\t%s' %i)
				idx = i
				prev = None
				while True:
					parent = parents[idx - 1]
					#print(parent)
					if parent == -1:
						break
					tree = Tree()
					if prev is not None:
						tree.add_child(prev)
					trees[idx - 1] = tree
					tree.idx = idx - 1
					if parent - 1 in trees.keys():
						trees[parent - 1].add_child(tree)
						break
					elif parent == 0:
						root = tree
						break
					else:
						prev = tree
						idx = parent
		return root