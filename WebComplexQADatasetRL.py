import torch
import torch.utils.data as data
import os
import re
#from nltk.corpus import stopwords
import json
from copy import deepcopy
import numpy as np

#stop_words = stopwords.words('english')
my_idx = 80000#[-1]
np.random.seed(123)

class WebComplexQADatasetRL(data.Dataset):
	def __init__(self, path, dic = None, limit = 1000, topic_limit = 450, is_train = False):
		super(WebComplexQADatasetRL, self).__init__()
		self.answers = self.read_answers(os.path.join(path, 'answer.txt')) #Enrichrelanswer
		self.sentences = self.read_sentences(os.path.join(path, 'nlq.txt'), dic)
		self.templates = self.read_templates(os.path.join(path, 'nlq_tem.txt'), dic)
		if is_train:
			self.to_entities, self.coincident = self.read_topic_entities(os.path.join(path, 'predict.json'), #filter_topicentity_all_types
														os.path.join(path, 'topicentity_all_types.json'), dic, topic_limit = topic_limit) # topicentity_all_types
		else:
			self.to_entities, self.coincident = self.read_topic_entities(os.path.join(path, 'predict.json'),
														os.path.join(path, 'topicentity_all_types.json'), dic, topic_limit = topic_limit) # topicentity_all_types_copy
		if topic_limit:
			self.cand_paths, self.cand_paths_score, self.cand_paths_an, self.cand_path_cont = self.read_candidate_paths(os.path.join(path,
				'candidate_paths.json'), dic, limit = limit, entities = self.to_entities) #candidate_paths_SMART
		else:
			self.cand_paths, self.cand_paths_score, self.cand_paths_an, self.cand_path_cont = self.read_candidate_paths(os.path.join(path,
				'candidate_paths.json'), dic, limit = limit) # candidate_paths_EnrichSMART
		self.to_entities = self.filter_topic_entities(self.cand_paths, self.cand_path_cont, self.to_entities)
		# print(len(self.sentences))
		# print(len(self.templates))
		# print(len(self.cand_paths))
		# print(len(self.cand_paths_score))
		# print(len(self.to_entities))

	def __getitem__(self, index):
		sentence = deepcopy(self.sentences[index])
		answer = deepcopy(self.answers[index])
		template = deepcopy(self.templates[index])
		cand_path = deepcopy(self.cand_paths[index])
		cand_path_score = deepcopy(self.cand_paths_score[index])
		cand_path_an = deepcopy(self.cand_paths_an[index])
		cand_path_cont = deepcopy(self.cand_path_cont[index])
		to_entity = deepcopy(self.to_entities[index])
		return (sentence, template, answer, cand_path, cand_path_score, cand_path_an, cand_path_cont, to_entity)

	def concate(self, other):
		self.sentences += other.sentences
		self.answers += other.answers
		self.templates += other.templates
		self.cand_paths += other.cand_paths
		self.cand_paths_score += other.cand_paths_score
		self.cand_paths_an += other.cand_paths_an
		self.cand_path_cont += other.cand_path_cont
		self.to_entities += other.to_entities

	def len(self):
		return len(self.sentences)

	def read_answers(self, filename):
		with open(filename, 'r', encoding = 'utf-8') as f: #
			answers = [self.read_answer(line) for line_idx, line in enumerate(f) if line_idx < my_idx]
		return answers

	def read_answer(self, line):
		#line = json.loads(line).strip().lower().split('|')
		line = re.split('\||\t', line.strip().lower())
		return line

	def read_sentences(self, filename, dic = None):
		with open(filename, 'r', encoding = 'utf-8') as f: #
			sentences = [self.read_sentence(line, dic) for line_idx, line in
				enumerate(f) if line_idx < my_idx]
		return sentences

	def read_sentence(self, line, dic):
		line = line.strip().lower()
		line = re.sub('\'s', ' s', line)
		words = []
		for w_idx, w in enumerate(line.split(' ')):
			w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
			if dic and w not in dic:
				dic[w] = len(dic)
			words += [w]
		line = ' '.join(words)
		return line

	def read_templates(self, filename, dic = None):
		with open(filename, 'r', encoding = 'utf-8') as f: #
			templates = [self.read_template(line, dic) for line_idx, line in
			enumerate(f) if line_idx < my_idx]
		return templates

	def read_template(self, line, dic):
		line = re.sub('[^a-z\<E\>]', ' ', line)
		line = re.sub(' +', ' ', line)
		line = line.strip(' ').lower()
		words = [w for w in line.split(' ')] # if w not in stop_words
		for w in words:
			if dic and w not in dic:
				dic[w] = len(dic)
		words = ' '.join(words)
		return words

	def read_candidate_paths(self, filename, dic = None, limit = 500, entities = None):
		candidate_paths = []
		candidate_paths_score = []
		candidate_paths_an = []
		candidate_path_cont = []
		with open(filename, 'r') as f:
			for line_idx, line in enumerate(f):
				if line_idx < my_idx:
					if entities:
						entity = set(entities[line_idx].keys())
						#entity = set([entities[line_idx][e][0] for e in entities[line_idx]])
						p, s, n, c = self.read_candidate_path(line, dic, limit = limit, entities=entity)
					else:
						p, s, n, c = self.read_candidate_path(line, dic, limit = limit)
					candidate_paths += [p]
					candidate_paths_score += [s]
					candidate_paths_an += [n]
					candidate_path_cont += [c]
		return candidate_paths, candidate_paths_score, candidate_paths_an, candidate_path_cont

	def read_candidate_path(self, line, dic, limit = 1000, entities=None):
		line = json.loads(line)
		candidate_path = []
		candidate_path_score = []
		candidate_paths_an = []
		candidate_paths_cont = []
		if entities:
			score_bound, limit = 0., limit
		else:
			score_bound, limit = 0., limit
		for R in line:
			r = R.lower()
			if len(candidate_path) < 2000:
				for t in r.split('\t'):
					if not re.search('^[mg]\.', t):
						if dic and t.split('.')[-1] not in dic:
							dic[t.split('.')[-1]] = len(dic)
						for w in re.split('\.|\_| ', t):
							if dic and w not in dic:
								dic[w] = len(dic)
				# print(r.split('\t')[0])
				# print(entities)
				# exit()
				# if entities and (r.split('\t')[0] not in entities):
				# 	continue

				if (line[R][0] > score_bound or np.random.random() < limit*1./len(line)):# and len(re.split('\.|\_|\t', r)) < 50:
					candidate_path += [r]
					candidate_path_score += [line[R][0]]
					if isinstance(line[R][1], int):
						candidate_paths_an += [line[R][1]]
					else:
						candidate_paths_an += [len(line[R][1])]
					candidate_paths_cont += [line[R][2]]
		# if np.max(candidate_path_score) != np.max([line[r][0] for r in line]):
		# 	print(line)
		# 	print(entities)
		# 	print(np.max(candidate_path_score))
		# 	print(np.max([line[r][0] for r in line]))
		# 	exit()
		#print(candidate_path_score)
		return candidate_path, candidate_path_score, candidate_paths_an, candidate_paths_cont

	def read_topic_entities(self, filename1, filename2, dic, topic_limit=5000):
		topic_entities = []
		coincident = {}
		with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
			for line_idx, line in enumerate(zip(f1, f2)):
				if line_idx < my_idx:
					e = self.read_topic_entity(line[0], line[1], dic, topic_limit=topic_limit)
					if len(e) == 0:
						e = {'m.03_dwn': (u'lou seal', u'0.875372', 0)}
					topic_entities += [e]
					for t in e:
						if t in coincident:
							coincident[t] |= (set(e) - set(t))
						else:
							coincident[t] = (set(e) - set(t))
					if line_idx % 100 == 0:
						print(line_idx)
		return topic_entities, coincident

	def read_topic_entity(self, line0, line1, dic, topic_limit=5000):
		line0 = json.loads(line0)
		line1 = json.loads(line1)
		#line1 = [i for i in line1 if np.random.random() < 0.5]
		line = line0 + line1
		#line = line[:topic_limit]
		topic_entity = {}
		for mid in line:
			if re.search('^[mg]\.', mid[0]):
				alias = re.split('\_| |\-', mid[1].lower())
			else:
				alias = re.split('\.|\_| |\/', mid[0].lower())[1: ]
			if (mid[3] > 0 or np.random.random() < topic_limit*1./len(line)) and len(topic_entity) < topic_limit: # np.random.random() < 50./len(line)
				words = []
				for w in alias:
					w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
					if dic and w not in dic:
						dic[w] = len(dic)
					for letter in w:
						if dic and letter not in dic:
							dic[letter] = len(dic)
					words += [w]
				words = ' '.join(words)
				if re.search('^[mg]\.', mid[0]):
					if len(re.split(' ', words)) > -1:
						#topic_entity[mid[1].lower()] = (words, mid[2], mid[3])
						topic_entity[mid[0]] = (words, mid[2], mid[3])
				else:
					if len(re.split('\.|\_| |\/', mid[0])) > -1:
						topic_entity[mid[0].lower()] = (mid[1], mid[2], mid[3])
		return topic_entity

	def filter_topic_entities(self, cand_paths, cand_path_conts, entities):
		new_entities = []
		for idx, cand_path in enumerate(cand_paths):
			cand_path_set = set(sum([re.split('\t', cs) for cs in cand_path], [])) | (set(sum(cand_path_conts[idx], [])) - set(['']))
			new_entity = {}
			for e_idx, e in enumerate(entities[idx]):
				if e in cand_path_set:
					new_entity[e] = entities[idx][e]
			new_entities += [new_entity]
		return new_entities
