import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data
from utils.tree import Tree
import os
import re

parsing_subs = ['dobj', 'pobj', 'iobj']
parsing_subs2 = ['det', 'nn', 'pobj', 'nsubj']
stop_words = ['what', 'when', 'which', 'where', 'when', 'how', 'with', 'in',
			'to', 'of', 'the', 'as', 'to', 'for', 'by', 'it', 'a', 'from',
			'that']
my_idx = -8

class ComplexWebQuestions(data.Dataset):
	def __init__(self, path):
		super(ComplexWebQuestions, self).__init__()
		self.sentences = self.read_sentences(os.path.join(path, 'nlq.toks'))
		self.trees = self.read_trees(os.path.join(path, 'nlq.parents'))
		self.dependencies = self.read_dependencies(os.path.join(path, 'nlq.rels'))

	def __getitem__(self, index):
		tree = deepcopy(self.trees[index])
		sent = deepcopy(self.sentences[index])
		return (tree, sent)

	def read_dependencies(self, filename):
		with open(filename, 'r') as f:
			dependencies = [self.read_dependency(line) for line_idx, line in
				enumerate(tqdm(f.readlines())) if not line_idx == my_idx]
		return dependencies

	def read_dependency(self, line):
		return line.strip().split()

	def read_sentences(self, filename):
		with open(filename, 'r') as f:
			sentences = [self.read_sentence(line) for line_idx, line in
				enumerate(tqdm(f.readlines())) if not line_idx == my_idx]
		return sentences

	def read_sentence(self, line):
		return line

	def read_trees(self, filename):
		with open(filename, 'r') as f:
			trees = [self.read_tree(line) for line_idx, line in
				enumerate(tqdm(f.readlines())) if not line_idx == my_idx]
		return trees

	def read_tree(self, line):
		parents = list(map(int, line.split()))
		trees = dict()
		root = None
		for i in range(1, len(parents) + 1):
			if i - 1 not in trees.keys() and parents[i - 1] != -1:
				idx = i
				prev = None
				while True:
					parent = parents[idx - 1]
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
		#root.visual()
		return root

	def parse_to_subjects(self):
		total_subjects = []
		#g = open(out_path, 'w')
		for idx, root in enumerate(self.trees):
			print(idx)
			dependency = self.dependencies[idx]
			sentence = self.sentences[idx].lower().split()
			# print(list(zip(list(range(len(dependency))), dependency)))
			# print(list(zip(list(range(len(sentence))), sentence)))
			#print(idx)
			# merge compound words
			subject = []
			#root.visual()
			try:
				self.find_subject2(root, dependency, sentence, subject)
			except:
				pass
			# print(subject)
			# exit()
			subject = self.clean_subjects(subject)
			subject = '|'.join(subject)
			total_subjects += [subject]
			# form logical form
			if idx % 1000 == 0:
				print('read trees %s ...' %idx)
				#exit()
		return total_subjects

	def find_subject(self, root, dependency, sentence, subject):
		for c in root.children:
			if dependency[c.idx] in parsing_subs:
				sub = [t.idx for t in c.all_children([])]+[c.idx]
				sub = sorted(list(set(sub)))
				subject += [sentence[s] for s in sub]
				#print(subject)
				break
			else:
				self.find_subject(c, dependency, sentence, subject)

	def find_subject2(self, root, dependency, sentence, subject):
		for c in root.children:
			c_idx = [s.idx for s in c.all_children([])]
			c_label = [dependency[s] for s in c_idx]
			# print(c.idx)
			# print(c_label)
			if set(c_label) <= set(parsing_subs2) and len(c_label) > 0:
				sub = c_idx + [c.idx]
				sub = sorted(list(set(sub)))
				subject += [[sentence[s] for s in sub]]
			else:
				self.find_subject2(c, dependency, sentence, subject)

	def clean_subjects(self, subjects):
		new_subjects = []
		for subject in subjects:
			if not set(subject) < set(stop_words):
				if subject[0] in stop_words:
					new_subjects += [' '.join(subject[1:])]
		return new_subjects

def FormArguments(data_path, arguments):
	g = open(os.path.join(data_path, 'nlq2sub.toks'), 'w')
	with open(os.path.join(data_path, 'nlq.toks'), 'r') as f:
		for line_idx, line in enumerate(f):
			line = line.rstrip().lower()
			if len(arguments[line_idx]) > 0:
				g.write('\t'.join([line, arguments[line_idx]]) + '\n')
	g.close()

def form_subject_via_POS(data_path):
	q_path = os.path.join(data_path, 'nlq.txt')
	questions = open(q_path).readlines()
	out_path = os.path.join(data_path, 'nlq2sub.toks')
	g = open(out_path, 'w')
	with open(os.path.join(data_path, 'nlq.tags')) as f:
		for line_idx, line in enumerate(f):
			line = line.strip('\n')
			line = line.split(' ')
			q = questions[line_idx].strip('\n').split(' ')

			tag, ntag, ttag = [], [], []
			for idx, t in enumerate(line):
				if t in ['NNP', 'NNPS', 'CD']:
					tag += [q[idx]]
				elif t in ['NN', 'NNS']:
					ntag += [q[idx]]
				else:
					tag += ['\t']
					ntag += ['\t']
				if idx > 0 and q[idx] not in ['', 'I'] and q[idx][0].isupper():
					ttag += [q[idx]]
				else:
					ttag += ['\t']
			# if line_idx == 524:
			# 	print(ttag)
			tag = clean_tag(tag)
			ntag = clean_tag(ntag)
			ttag = clean_tag(ttag)
			# if line_idx == 2:
			# 	print(q)
			# 	print(tag)
			# 	exit()
			g.write('|'.join(list(set(tag + ttag))) + '|\t|')
			g.write('|'.join(ntag) + '\n')
	g.close()

def form_subject_via_constituency_path(data_path):
	def toTree(expression):
		tree = dict()
		msg =""
		stack = list()
		for char in expression:
			if(char == '('):
				stack.append(msg)
				msg = ""
			elif char == ')':
				parent = stack.pop()
				if parent not in tree:
					tree[parent] = list()
				tree[parent].append(msg)
				msg = parent
			else:
				if msg == "":
					msg += char
				else:
					msg += ('\t' + char)
		return tree
	def displayTree(d, dic, tokens):
		for w in dic[d]:
			if len(w.split('\t')) == 1:
				displayTree(w, dic, tokens)
			else:
				tokens += [w]

	out_path = os.path.dirname(data_path)
	g = open(os.path.join(out_path, 'nlq2sub.toks'), 'w')
	with open(data_path) as f:
		for line_idx, line in enumerate(f):
			if line_idx > -1:
				line = line.strip()
				line = line.replace('(', '( ').replace(')', ' )')
				line = re.sub(' +', ' ', line)
				line = line.split(' ')
				line = [str(i)+'|'+w if w not in ['(', ')'] else w for i,w in enumerate(line)]
				tree = toTree(line)
				tokens = []
				for t in tree:
					if t != '' and re.sub('^[0-9]+\|', '', t) in ['NP']:
						token = []
						displayTree(t, tree, token)
						tokens += [' '.join([t.split('|')[-1] for t in token])]
				g.write('|'.join(tokens) + '\n')
				if line_idx % 100 == 0:
					print('form_subject_via_constituency_path %s' %line_idx)
	g.close()

def clean_tag(tag):
	tag = ' '.join(tag)
	tag = re.sub('(\t \t)+', '\t', tag).split('\t')
	tag = [t.strip() for t in tag if t not in ['', ' ']]
	tag = [re.sub('[\".\:\',]$|^[\".\:\',]', '', t) for t in tag]
	return tag

if __name__ == '__main__':
	base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	data_path = os.path.join(base_path, 'data')
	train_path = os.path.join(data_path, 'train')
	dev_path = os.path.join(data_path, 'dev')
	test_path = os.path.join(data_path, 'test')

	# train_dataset = ComplexWebQuestions(train_path)
	# train_arguments = train_dataset.parse_to_subjects()
	# FormArguments(train_path, train_arguments)
	#
	# dev_dataset = ComplexWebQuestions(dev_path)
	# dev_arguments = dev_dataset.parse_to_subjects()
	# FormArguments(dev_path, dev_arguments)
	#
	# test_dataset = ComplexWebQuestions(test_path)
	# test_arguments = test_dataset.parse_to_subjects()
	# FormArguments(test_path, test_arguments)

	#form_subject_via_POS(dev_path)
	#form_subject_via_POS(train_path)
	#form_subject_via_POS(test_path)

	form_subject_via_constituency_path(os.path.join(dev_path, 'nlq.cpath'))
	form_subject_via_constituency_path(os.path.join(train_path, 'nlq.cpath'))
	form_subject_via_constituency_path(os.path.join(test_path, 'nlq.cpath'))
