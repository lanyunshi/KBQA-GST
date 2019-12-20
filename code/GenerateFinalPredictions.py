import json
import os
import random
import re
import pickle
import numpy as np
import copy
import eval
from utils.GenerateCandidatePathViaSparqlPool import *
from collections import defaultdict
from nltk.util import ngrams
from nltk.corpus import stopwords

stop_word = set('''nation country state university instrument province continent airport high school county
                book citytown cancer flower baseball tv rainforest guitar bird island'''.split())
#stop_word = set('''country college country team lagnuage state '''.split())
#word_vectors = api.load("glove-wiki-gigaword-100")

def generate_final_predictions(pred, folder, category, threshold = 0.9, use_type = True):
    test_path = os.path.join('data', '%s_%s' %(category, folder))
    saved_path = os.path.join('saved-model', folder)

    pred_file = 'pred%s_test10.txt' %pred
    out_path = os.path.join(saved_path, 'final_predictions%s.json' %pred)

    lineidx2qidx = {}
    if 'CWQ' in folder:
        with open('data/CWQ/ComplexWebQuestions_%s.json' %category) as f:
            lines = json.load(f)
            for line_idx, line in enumerate(lines):
                lineidx2qidx[line_idx] = line["ID"]
    elif 'WBQ' in folder:
        with open('data/WBQ/qids.json') as f:
            lines = json.load(f)
            for line_idx, line in enumerate(lines):
                lineidx2qidx[line_idx] = lines[line_idx]

    answers, content = [], {}
    if 'CWQ' in saved_path:
        with open('data/CWQ/ComplexWebQuestions_test_wans.json') as f:
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
        with open('data/%s_CWQ/candidate_paths_EnrichSMART2.json' %category) as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                candidate_path_pool += [line]
    elif 'WBQ' in saved_path:
        with open(os.path.join(test_path, 'answer.txt')) as f:
            for line_idx, line in enumerate(f):
                line = line.strip().lower()
                answers += [line.split('\t')]
        candidate_path_pool = [] # if 'Enrich' in pred else 'candidate_paths_SMART.json'
        #if os.path.exists(os.path.join(test_path, 'candidate_paths_EnrichSMART2.json')):
        with open(os.path.join(test_path, 'candidate_paths_EnrichSMART2.json')) as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                candidate_path_pool += [line]
                # for r in line:
                #     content[r] = line[r][1]

    if use_type:
        nlqs = []
        with open(os.path.join(test_path, 'nlq.txt')) as f:
            for line_idx, line in enumerate(f):
                line = line.strip().lower()
                line = re.sub('\<e\>', ' ', line)
                line = re.sub('[ ]+', ' ', line)
                nlqs += [line]

    preds, precision = [], 0.
    mid2namef = open('data/mid2name.json')
    mid2name = json.load(mid2namef)

    with open(os.path.join(saved_path, pred_file)) as f2:
        for line_idx, line in enumerate(f2):
            if line_idx > -1:
                pred, final_anstxt, anstxt = {}, [], ''
                line = line.strip()

                if not re.search('[a-z]', line): continue
                if len(line.split(' | ')) == 2:
                    topic_entity, r = line.split(' | ')
                else:
                    r = '\t'.join(line.split('\t')[1:])

                rs = re.split('\|[\d\.]+', r)
                rs = [re.sub('^\t', '', rr) for rr in rs]
                ss = re.findall('(?<=\|)[\d\.]+', r)

                ans, seen = [], {}
                if len(content) == 0:
                    #print(candidate_path_pool[line_idx])
                    if 'SQ' in saved_path:
                        for r_idx, r in enumerate(rs[:1]):
                            for p in candidate_path_pool[line_idx]:
                                if p.lower().encode('ascii', 'ignore').decode('ascii') == r.lower():
                                    for a in candidate_path_pool[line_idx][p][1]:
                                        ans += [[a.lower(), float(ss[r_idx])]]
                    else:
                        if 'WBQ' in folder: rs = rs[:1]
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
                    pred_rel = relanswer[line_idx] if relanswer[line_idx] in rs else rs[0]
                    for r_idx, r in enumerate(rs[:1]):
                        if r in content:
                            if r in mid2name:
                                anss = mid2name[r]
                            else:
                                try:
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
                                if a not in seen:
                                    seen[a] = len(ans)
                                    ans += [[a, float(ss[r_idx])]]
                                else:
                                    continue
                                    ans[seen[a]][1] += float(ss[r_idx])

                anstxt = []
                for a, s in ans:
                    if 'CWQ' in saved_path:
                        if a in mid2name:
                            anss = mid2name[a]
                        else:
                            try:
                                anss = mid2name_query(a)[0]
                                mid2name[a] = anss
                            except:
                                anss = ""
                    else:
                        anss = a
                    anstxt += [[anss, s]]
                # print(anstxt)
                # print(answers[line_idx])
                # exit()

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
                                names = mid2name_query(a)
                                types = type_query(a)
                                types = ' '.join(names + types)
                                mid2name['*'+a] = types
                            except:
                                types = [""]
                        types = types[0] if isinstance(types, list) else types
                        #print(types)
                        sim = constraint_detection(nlqs[line_idx].lower().split(), types.split())
                        anstxt1 += [[a, sim]]
                    anstxt = copy.deepcopy(anstxt1)

                if len(anstxt)> 0:
                    #top_score = sorted(anstxt1, key=lambda x: x[1])[::-1][:1][0][1]
                    top_score = np.max([0., sorted(anstxt, key=lambda x: x[1])[::-1][:1][0][1]])
                    anstxt1.sort(key=lambda x: x[1], reverse = True)
                    for a in anstxt:
                        if a[1] >= top_score:# and a[1] > 0.5:
                            final_anstxt += [a[0]]
                #print(final_anstxt)

                final_anstxt = final_anstxt if len(final_anstxt) > 0 else ['']
                if 'CWQ' in folder: final_anstxt = final_anstxt[:1]
                #print(final_anstxt, set(answers[line_idx][0])); exit
                if len(set(answers[line_idx][0]) & set(final_anstxt)) > 0:
                #if r in relanswer[line_idx]:
                    precision += 1
                else:
                    pass# if line_idx % 500 == 0:

                if 'CWQ' in saved_path:
                    pred["ID"] = lineidx2qidx[line_idx]
                    pred["answer"] = final_anstxt
                else:
                    pred["QuestionId"] = lineidx2qidx[line_idx]
                    pred["Answers"] = final_anstxt
                preds += [pred]

                if line_idx % 1000 == 0:
                    print('process %s ...' %line_idx)

    print('unofficial precision %s' %(precision/len(preds)))
    g = open(out_path, 'w')
    json.dump(preds, g)
    g.close()
    return precision*1./len(preds)

def compute_f1(p, t):
    precision = len(set(p)&set(t))*1. / np.max([len(p), 1e-10])
    recall = len(set(p)&set(t))*1. / np.max([len(t), 1e-10])
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
    #print(q)
    if len(set(q) & set(r)) > 0:
        for i in range(np.min([len(r), len(q)]), 0, -1):
            q1 = ngrams(q, i)
            q2 = ngrams(r, i)
            if len(set(q1) & set(q2)) > 0:
                sim = i
                if sim > 0:
                    return sim
    return 0.

def main(pred):
    folder = 'WBQ'
    for category in ['test']:
        foldername = os.path.dirname(os.getcwd())
        for threshold in [1]:
            generate_final_predictions(pred, folder, category, threshold = threshold) #

if __name__ == "__main__":
    main('TG+EE_tmp')
