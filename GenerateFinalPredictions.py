import json
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import random
import re
import pickle
import numpy as np
import copy
import eval
from collections import defaultdict
from nltk.util import ngrams
# import gensim.downloader as api
from nltk.corpus import stopwords

#stop_word = set(stopwords.words('english'))
# stop_word = stop_word.union(set(('the').split()))
stop_word = set('''nation country state university instrument province continent airport high school county book citytown cancer flower baseball tv rainforest
				 guitar bird island'''.split())
#stop_word = set('''country college country team lagnuage state '''.split())
#word_vectors = api.load("glove-wiki-gigaword-100")

def generate_final_predictions(pred, test_path, saved_path, use_snippet = False, threshold = 0.9, use_type = True):
	folder_path = os.path.dirname(test_path)
	category = os.path.basename(test_path)
	#pred = 'EE_GoldenRel'
	pred_file = 'pred%s_test10.txt' %pred
	out_path = os.path.join(saved_path, 'final_predictions%s.json' %pred)
	#out_path = os.path.join(saved_path, 'entity_predictions%s.json' %pred)

	lineidx2qidx = {}
	if '2018-11-27-11-42-59RL' in saved_path:
		with open(os.path.join(folder_path, 'ComplexWebQuestions/ComplexWebQuestions_%s.json' %category)) as f:
			lines = json.load(f)
			for line_idx, line in enumerate(lines):
				lineidx2qidx[line_idx] = line["ID"]
	elif False:#'2018-11-29-16-01-25RL' not in saved_path:
		qids_folder = 'WebQuestionsSP' if '2018-11-20-02-20-31RL' in saved_path else 'WikiMovie'
		with open(os.path.join(folder_path, '%s/qids.json' %qids_folder)) as f:
			lines = json.load(f)
			for line_idx, line in enumerate(lines):
				lineidx2qidx[line_idx] = lines[line_idx]

	if use_snippet:
		if '2018-11-27-11-42-59RL' not in saved_path:
			qidx2entity = defaultdict(set)
			snippet_folder = 'WBQ' if '2018-11-20-02-20-31RL' in saved_path else 'WikiMovie'
			with open('/home/yunshi/Dropbox/MnemonicReader/data/datasets/%s-test.json' %snippet_folder) as f:
				lines = json.load(f)
			for paragraph in lines['data']:
				for para in paragraph['paragraphs']:
					for qas in para['qas']:
						qidx = qas['id']
						qidx2entity[qidx] = qidx2entity[qidx].union(set([(a['text'].lower(), a['mid']) for a in qas['answers']]))
		snippet_preds = []
		with open(use_snippet) as f:
			lines = json.load(f)
		for line_idx in range(len(lineidx2qidx)):
			if lineidx2qidx[line_idx] in lines:
				snippet_pred = {}
				#print(line_idx)
				#print(lines[lineidx2qidx[line_idx]])
				#print(qidx2entity[lineidx2qidx[line_idx]])
				for span, score in lines[lineidx2qidx[line_idx]]:
					span = span.lower()
					if '2018-11-27-11-42-59RL' not in saved_path:
						span_old = span
						for e, m in qidx2entity[lineidx2qidx[line_idx]]:
							if e in span and '2018-11-20-02-20-31RL' in saved_path:
								span = m
							elif e in span and '2018-10-31-12-02-25RL' in saved_path:
								span = e
								span_old = '#'+span_old
					if '2018-11-27-11-42-59RL' in saved_path or span_old != span:
						if span not in snippet_pred:
							snippet_pred[span] = score
						else:
							#if score > snippet_pred[span]:
							snippet_pred[span] += score
				# print(snippet_pred)
				# if line_idx > 99:
				# 	exit()
			else:
				snippet_pred = None
			snippet_preds += [snippet_pred]
		#print(len(snippet_preds))
		# g = open('/home/yunshi/Dropbox/MnemonicReader/data/predict/snippet_preds2.json', 'w')
		# json.dump(snippet_preds, g)
		# g.close()
		#exit()
	# f = open('/home/yunshi/Dropbox/MnemonicReader/data/predict/snippet_preds2.json')
	# snippet_preds = json.load(f)
	# f.close()

	answers = []
	content = {}
	if '2018-11-27-11-42-59RL' in saved_path:
		with open(os.path.join(folder_path, 'ComplexWebQuestions/ComplexWebQuestions_test_wans.json')) as f:
			lines = json.load(f)
			for line_idx, line in enumerate(lines):
				answer = []
				for a in line["answers"]:
					if "aliases" in a:
						answer += a["aliases"] + [a["answer"]]
					else:
						answer += [a["answer"]]
				answers += [([ans.lower() for ans in answer if ans != None], len(line["answers"]))]
		candidate_path_pool = []
		with open(os.path.join(folder_path, '%s/candidate_paths_EnrichSMART2.json' %category)) as f:
			for line_idx, line in enumerate(f):
				line = json.loads(line)
				candidate_path_pool += [line]
	elif '2018-11-20-02-20-31RL' in saved_path:
		with open('%s_WBQ/answer.txt' %test_path) as f:
			for line_idx, line in enumerate(f):
				line = line.strip().lower()
				answers += [line.split('\t')]
		relanswer = []
		with open('%s_WBQ/Enrichrelanswer.json' %test_path) as f:
			for line_idx, line in enumerate(f):
				line = json.loads(line)
				relanswer += [line]
		candidate_path_file = 'candidate_paths_EnrichSMART2.json'# if 'Enrich' in pred else 'candidate_paths_SMART.json'W
		if os.path.exists('data/test_WBQ/%s' %candidate_path_file):
			with open('data/test_WBQ/%s' %candidate_path_file) as f:
				for line_idx, line in enumerate(f):
					line = json.loads(line)
					for r in line:
						content[r] = line[r][1]
	elif '2019-02-05-09-12-52RL' in saved_path:
		lineidx2qidx = {}
		with open('%s_CQ/answer.txt' %test_path) as f:
			for line_idx, line in enumerate(f):
				line = line.strip().lower()
				answers += [line.split('|')]
				lineidx2qidx[line_idx] = line_idx
		candidate_path_file = 'candidate_paths.json'# if 'Enrich' in pred else 'candidate_paths_SMART.json'W
		if os.path.exists('data/test_CQ/%s' %candidate_path_file):
			with open('data/test_CQ/%s' %candidate_path_file) as f:
				for line_idx, line in enumerate(f):
					line = json.loads(line)
					for r in line:
						content[r] = line[r][1]
	elif '2018-10-31-12-02-25RL' in saved_path:
		with open(os.path.join(folder_path, '%s_WikiMovie/answer.txt' %category)) as f:
			for line_idx, line in enumerate(f):
				line = line.strip().lower()
				answer = line.split('|')
				answers += [answer]
		candidate_path_pool = []
		with open(os.path.join(folder_path, '%s_WikiMovie/candidate_paths.json' %category)) as f:
			for line_idx, line in enumerate(f):
				line = json.loads(line)
				candidate_path_pool += [line]
	elif '2018-11-29-16-01-25RL' in saved_path:
		lineidx2qidx = {}
		with open(os.path.join(folder_path, '%s_SQ/relanswer.json' %category)) as f:
			for line_idx, line in enumerate(f):
				line = json.loads(line)
				answers += [line]
				lineidx2qidx[line_idx] = line_idx
		candidate_path_pool = []
		with open(os.path.join(folder_path, '%s_SQ/candidate_paths_EnrichSMART.json' %category)) as f:
			for line_idx, line in enumerate(f):
				line = json.loads(line)
				candidate_path_pool += [line]

	if use_type:
		nlqs = []
		#word_vectors = api.load("glove-wiki-gigaword-100")
		filename = '%s_CQ/nlq.txt' %test_path if '2019-02-05-09-12-52RL' in saved_path else '%s_WBQ/nlq.txt' %test_path
		with open(filename) as f:
			for line_idx, line in enumerate(f):
				line = line.strip().lower()
				line = re.sub('\<e\>', ' ', line)
				line = re.sub('[ ]+', ' ', line)
				nlqs += [line]

	preds = []
	topic_entity_num = []
	mid2namef = open('data/mid2name.json')
	mid2name = json.load(mid2namef)

	precision = 0.
	# sparql_list = [f[:-4] for f in os.listdir('data/freebase/cache')]

	with open(os.path.join(saved_path, pred_file)) as f2:
		for line_idx, line in enumerate(f2):
			if line_idx > -1:
				pred = {}
				line2 = line
				line2 = line2.strip()
				anstxt = ''
				final_anstxt = []
				if re.search('[a-z]', line2):
					if True:
						topic_entity, r = line2.split(' | ')
						# topic_entities = topic_entity.split('\t')[-1]
						# topic_entity_num += [len(set(re.findall('[mg]\.|1|2', topic_entities)))]
						# if line_idx < 10:
						# 	print(re.findall('[mg]\.|1|2', topic_entities))
						# 	print(len(set(re.findall('[mg]\.|1|2', topic_entities))))
						#r = '\t'.join(line2.split('\t')[1:])
						rs = re.split('\|[\d\.]+', r)
						rs = [re.sub('^\t', '', rr) for rr in rs]
						ss = re.findall('(?<=\|)[\d\.\-e]+', r)

						ans = []
						seen = {}
						if len(content) == 0:
							#print(candidate_path_pool[line_idx])
							if '2018-10-31-12-02-25RL' in saved_path:
								for r_idx, r in enumerate(rs[:1]):
									for p in candidate_path_pool[line_idx]:
										if p.lower().encode('ascii', 'ignore').decode('ascii') == r.lower():
											for a in candidate_path_pool[line_idx][p][1]:
												ans += [[a.lower(), float(ss[r_idx])]]
							else:
								for r_idx, r in enumerate(rs):
									for p in candidate_path_pool[line_idx]:
										if p.lower() == r.lower():
											for a in candidate_path_pool[line_idx][p][1]:
												if a not in seen:
													seen[a] = len(ans)
													ans += [[a, float(ss[r_idx])]]
												else:
													ans[seen[a]][1] += float(ss[r_idx])
												#break
						else:
							ans_score = defaultdict(float)
							#pred_rel = relanswer[line_idx] if relanswer[line_idx] in rs else rs[0]
							for r_idx, r in enumerate(rs[:1]):
								if r in content:
									if r in mid2name:
										anss = mid2name[r]
									#anss = set(content[r])
									else:
										try:
											#print(r)
											if len(r.split('\t')) == 2:
												anss = hop1_rel_query(r)
											elif len(r.split('\t')) == 3:
												anss = hop2_rel_query(r)
											elif len(r.split('\t')) ==4:
												anss = hop3_rel_query(r)
											elif len(r.split('\t')) ==5:
												anss = hop4_rel_query(r)
										except:
											anss = [""]
										mid2name[r] = anss
									for a in anss:
										if r_idx == 0:
											if a not in seen:
												seen[a] = len(ans)
												ans += [[a, float(ss[r_idx])]]
											else:
												continue
												ans[seen[a]][1] += float(ss[r_idx])
										ans_score[a] += float(ss[r_idx])
						#print(ans_score)

						anstxt = []
						for a, s in ans:
							if '2018-11-27-11-42-59RL' in saved_path:# or '2019-02-05-09-12-52RL' in saved_path:
								if a in mid2name:
									anss = mid2name[a]
								else:
									try:
										anss = single_query(a)[0]
										mid2name[a] = anss
									except:
										anss = ""
							else:
								anss = a
							anstxt += [[anss, s]]
						#print(anstxt)

						if use_snippet:
							new_ans = copy.deepcopy(anstxt)
							anstxt_add = []
							seen = []
							snippet_pred = snippet_preds[line_idx]
							if snippet_pred:
								for s_idx, span_score in enumerate(new_ans):
									a, s = span_score
									if a in snippet_pred:
										#print('yeah')
										new_ans[s_idx][1] = threshold*new_ans[s_idx][1] + (1.-threshold)*snippet_pred[a]
										seen += [a]
									else:
										new_ans[s_idx][1] = threshold*new_ans[s_idx][1]
								for a in (set(snippet_pred) - set(seen)):
									if snippet_pred[a] > 0.:
										anstxt_add += [[a, (1.-threshold)*snippet_pred[a]]]
							anstxt1 = anstxt_add + new_ans#
						else:
							anstxt1 = anstxt
						anstxt = copy.deepcopy(anstxt1)

						if use_type:
							anstxt1 = []
							for a, s in anstxt[:1000]:
								if '|'.join(['*' + a, nlqs[line_idx]]) in mid2name:
									del mid2name['|'.join(['*' + a, nlqs[line_idx]])]
								if '*'+a in mid2name:
									types = mid2name['*'+a]
								else:
									try:
										names = single_query(a)
										types = type_query(a)
										types = ' '.join(names + types)
										mid2name['*'+a] = types
									except:
										types = [""]
								types = types[0] if isinstance(types, list) else types
								#print(types)
								sim = constraint_detection(nlqs[line_idx].lower().split(), types.split())
	 							if '2019-02-05-09-12-52RL' in saved_path:
									if a in mid2name:
										anss = mid2name[a]
									else:
										try:
											anss = single_query(a)[0]
											mid2name[a] = anss
										except:
											anss = ""
								else:
									anss = a
								anstxt1 += [[anss, ans_score[a]]] # + sim
							anstxt = copy.deepcopy(anstxt1)
							# print(anstxt)
							# exit()

						if len(anstxt)> 0:
							#top_score = sorted(anstxt1, key=lambda x: x[1])[::-1][:1][0][1]
							top_score = np.max([0., sorted(anstxt, key=lambda x: x[1])[::-1][:1][0][1]])
							anstxt1.sort(key=lambda x: x[1], reverse = True)
							for a in anstxt:
								if a[1] >= top_score:# and a[1] > 0.5:
									final_anstxt += [a[0]]

				final_anstxt = final_anstxt if len(final_anstxt) > 0 else ['']
				final_anstxt = final_anstxt#[:1]

				f1 = compute_f1(answers[line_idx], final_anstxt)
				if line_idx < 10:
					print(f1)
				#if f1 == 1:
				#if final_anstxt[0].encode('utf-8') in answers[line_idx]:
				#if rs[0].split('\t')[-1][:2] == relanswer[line_idx].split('\t')[-1][:2]:
				precision += f1

				if '2018-11-27-11-42-59RL' in saved_path:
					pred["ID"] = lineidx2qidx[line_idx]
					pred["answer"] = final_anstxt
				else:
					pred["QuestionId"] = lineidx2qidx[line_idx]
					pred["Answers"] = final_anstxt
				preds += [pred]

				if line_idx % 1000 == 0:
					print('process %s ...' %line_idx)

	print('unofficial precision %s' %(precision/len(preds)))
	#print('average topic entities %s' %(np.sum(topic_entity_num)*1./len(topic_entity_num)))
	if line_idx > 1500:
		g = open(out_path, 'w')
		json.dump(preds, g)
		g.close()
	# if precision*1./len(preds) > 0.365:
	# g = open('data/mid2name.json', 'w')
	# json.dump(mid2name, g)
	# g.close()
	return precision*1./len(preds)

def compute_f1(p, t, num = None):
	precision = len(set(p)&set(t))*1. / np.max([len(p), 1e-10])
	if num is None:
		recall = len(set(p)&set(t))*1. / np.max([len(t), 1e-10])
	else:
		recall = len(set(p)&set(t))*1. / np.max([num, 1e-10])
	f1 = 2.*precision*recall/np.max([(precision + recall), 1e-10])
	return f1

def constraint_detection(q, r):
	q = ' '.join(q)
	q = re.sub('ies ', 'y ', q)
	q = re.sub('s ', ' ', q)
	q = re.sub('college', 'college university', q)
	q = re.sub('highschool', 'high school', q)
	q = re.sub('city', 'city citytown', q)
	q = [w for w in q.split() if w in stop_word]
	r = [w for w in r if w in stop_word]
	if len(set(q) & set(r)) > 0:
		for i in range(np.min([len(r), len(q)]), 0, -1):
			q1 = ngrams(q, i)
			q2 = ngrams(r, i)
			if len(set(q1) & set(q2)) > 0:
				sim = i
				if sim > 0:
					return sim
	return 0.

def single_query(mid):
	ss = "ns:%s" %mid
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x1 WHERE {
%s ns:type.object.name ?x1
} limit 1
	""" %ss)
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	#print(results["results"]["bindings"])
	result = []
	for an in results["results"]["bindings"]:
		result += [an["x1"]["value"].lower()]
	return result

def rel_query(mid):
	ss = "ns:%s" %mid
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT distinct ?x1 WHERE {
%s ns:common.topic.description ?x1
} limit 1
	""" %ss)
	sparql.setReturnFormat(JSON)
	try:
		results = sparql.query().convert()
		#print(results["results"]["bindings"])
		result = results["results"]["bindings"][0]["x1"]["value"].lower()
	except:
		result = ''
	return result

def type_query(mid):
# %s ns:common.topic.notable_types ?x1.
# ?x1 ns:type.object.name ?x2
	ss = "ns:%s" %mid
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x2 WHERE {
%s ns:type.object.type ?x2
}
	""" %ss)
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	result = []
	for an in results["results"]["bindings"]:
		#result += [re.sub('\/', ' ', an["x2"]["value"].lower())]
		result += [an["x2"]["value"].lower().split('/')[-1]]
	result = [an.split('.')[-1] for an in result if an.split('.')[0] not in ['base', 'common', 'symbols']]
	result = [re.sub('\/|\_', ' ', ' '.join(result))]
	return result

def mid_query(mid):
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x1 WHERE {
?x1 ns:type.object.name \"%s\"@en
}
	""" %mid)
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	result = []
	for an in results["results"]["bindings"]:
		result += [an["x1"]["value"].split('/')[-1]]
	return result

def hop1_rel_query(r):
	h, r = r.split('\t')
	h = "ns:%s" %h
	r = "ns:%s" %r
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x1 WHERE {
%s %s ?x1
}
	""" %(h, r))
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	result = []
	for an in results["results"]["bindings"]:
		result += [re.sub('http://rdf.freebase.com/ns/', '', an["x1"]["value"])]
	return result

def hop2_rel_query(r):
	h, r1, r2 = r.split('\t')
	h = "ns:%s" %h
	r1 = "ns:%s" %r1
	r2 = "ns:%s" %r2
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x1 WHERE {
%s %s ?x2.
?x2 %s ?x1
}
	""" %(h, r1, r2))
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	result = []
	for an in results["results"]["bindings"]:
		result += [re.sub('http://rdf.freebase.com/ns/', '', an["x1"]["value"])]
	return result

def hop3_rel_query(r):
	h, r1, r2, t = r.split('\t')
	h = "ns:%s" %h
	r1 = "ns:%s" %r1
	r2 = "ns:%s" %r2
	t = "ns:%s" %t
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x1 WHERE {
%s %s ?x1.
?x1 %s %s.
}
	""" %(h, r1, r2, t))
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	result = []
	for an in results["results"]["bindings"]:
		result += [re.sub('http://rdf.freebase.com/ns/', '', an["x1"]["value"])]
	return result

def hop4_rel_query(r):
	h, r1, r2, r3, t = r.split('\t')
	h = "ns:%s" %h
	r1 = "ns:%s" %r1
	r2 = "ns:%s" %r2
	r3 = "ns:%s" %r3
	t = "ns:%s" %t
	sparql = SPARQLWrapper("http://localhost:8890/sparql")
	sparql.setQuery("""
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT ?x1 WHERE {
%s %s ?x2.
?x2 %s ?x1.
?x2 %s %s
}
	""" %(h, r1, r2, r3, t))
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	result = []
	for an in results["results"]["bindings"]:
		result += [re.sub('http://rdf.freebase.com/ns/', '', an["x1"]["value"])]
	return result

def main(pred):
	folder = '2019-02-05-09-12-52RL'
	for category in ['test']:
		foldername = os.path.dirname(os.getcwd())
		for threshold in [1]:
			print(threshold)
			final_eval = generate_final_predictions(pred, os.path.join('data', '%s' %category),
				'saved-model/%s' %folder,
				use_snippet = False, threshold = threshold) #
			if '2018-11-20-02-20-31RL' in folder:
				max_hit1 = []
				for _ in range(100):
					final_eval = eval.main('data/WebQuestionsSP/WebQSP.test.json', 'saved-model/2018-11-20-02-20-31RL/final_predictions%s.json' %pred)
					max_hit1 += [final_eval]
				print('max hit1 %s' %(np.max(max_hit1)))
	return final_eval
	#'%s/MnemonicReader/data/predict/ComplexWebQuestions-test-my_model4_with_T.mdl.preds' %foldername
	# '%s/MnemonicReader/data/predict/WikiMovie-test-WikiMovie_my_model_nokb.mdl.preds' %foldername 0.3
	#'%s/MnemonicReader/data/predict/WBQ-test-WBQ_my_model5_with_T.mdl.preds' %foldername
	#'/home/yunshi/Dropbox/AnswerRe-rank/saved-model/2018_10_30_09_57_27/predict.json'
	#'%s/MnemonicReader/data/predict/WikiMovie-test-WikiMovie_my_model_nokb.mdl.preds' %foldername
	# generate_entity_predictions(os.path.join('data', '%s' %category),
	# 	'saved-model/2018-11-27-11-42-59RL')
	# generate_final_predictions_with_final_entity(os.path.join('data', '%s' %category),
	# 	'saved-model/2018-11-27-11-42-59RL', use_snippet = '%s/MnemonicReader/data/predict/ComplexWebQuestions-test-20180817-d2326506.preds' %foldername)

if __name__ == "__main__":
    main('TG+EE_Emb200')
