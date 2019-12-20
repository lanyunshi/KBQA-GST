from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import re
import time
import pickle
import os

SPARQL_PATH = "http://10.0.104.43:8890/sparql" # Change to the path to your database

# when topic entity is mid and another is mid
mid_mid_sparql = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT DISTINCT ?out WHERE{
{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
?x4 ?y4 ?x2.
FILTER(?x0 != ?x4).
FILTER(?x0 != ?x2).
OPTIONAL{?x1 ns:type.object.name ?name1}.
OPTIONAL{?x2 ns:type.object.name ?name2}.
OPTIONAL{?x4 ns:type.object.name ?name4}.
FILTER(BOUND(?name2) && !BOUND(?name1) && BOUND(?name4) && regex(str(?x4), "ns/m\\\.|ns/g\\\.") && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
} UNION {
?x0 ?y2 ?x2.
?x4 ?y4 ?x2.
FILTER(?x0 != ?x4).
FILTER(?x0 != ?x2).
OPTIONAL{?x2 ns:type.object.name ?name2}.
OPTIONAL{?x4 ns:type.object.name ?name4}.
FILTER(BOUND(?name2) && BOUND(?name4) && regex(str(?x4), "ns/m\\\.|ns/g\\\.") && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
}
VALUES ?x0 {%s}.
VALUES ?x4 {%s}.
FILTER(!isLiteral(?x0) OR lang(?x0) = '' OR langMatches(lang(?x0), 'en')).
BIND(
IF(BOUND(?x1),
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x4),'\t',str(?y4),'\t',str(?x2)),
CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x4),'\t',str(?y4),'\t',str(?x2))) AS ?out)
}
'''

# when topic entity is mid and another is relation
mid_relation_sparql = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT * {
{SELECT DISTINCT ?out WHERE{
?x0 ?y2 ?x2.
{?x4 ?y4 ?x2.
BIND("1" AS ?z).
} UNION {
?x2 ?y4 ?x4}
FILTER(?x0 != ?x4).
FILTER(?x0 != ?x2).
OPTIONAL{?x2 ns:type.object.name ?name2}.
OPTIONAL{?x4 ns:type.object.name ?name4}.
FILTER(BOUND(?name2) && BOUND(?name4) && regex(str(?x4), "ns/m\\\.|ns/g\\\.") && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
FILTER(?y2 = %s || ?y4 = %s)
VALUES ?x0 {%s}.
BIND(
IF(BOUND(?z),
CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x4),'\t',str(?y4),'\t',str(?x2)),
CONCAT(str(?x0),'\t',str(?y2),'\t',str(?y4),'\t',str(?x4))) AS ?out)
}} UNION {
SELECT DISTINCT ?out WHERE{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
{?x4 ?y4 ?x2.
BIND("1" AS ?z).
} UNION {
?x2 ?y4 ?x4}
FILTER(?x0 != ?x4).
FILTER(?x0 != ?x2).
OPTIONAL{?x2 ns:type.object.name ?name2}.
OPTIONAL{?x1 ns:type.object.name ?name1}.
OPTIONAL{?x4 ns:type.object.name ?name4}.
FILTER(BOUND(?name2) && BOUND(?name4) && !BOUND(?name1) && regex(str(?x1), "ns/m\\\.|ns/g\\\.") && regex(str(?x4), "ns/m\\\.|ns/g\\\.") && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
FILTER(?y1 = %s || ?y2 = %s || ?y4 = %s)
VALUES ?x0 {%s}.
BIND(
IF(BOUND(?z),
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x4),'\t',str(?x4),'\t',str(?y4),'\t',str(?x2)),
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x4),'\t',str(?y4),'\t',str(?x4))) AS ?out)}}
}
'''

# when topic entity is mid and another is type
mid_type_sparql = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT * {
{SELECT DISTINCT ?out WHERE{
?x0 ?y2 ?x2.
?x2 ns:type.object.type ?x3.
FILTER(?x0 != ?x2).
OPTIONAL{?x2 ns:type.object.name ?name2}.
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
VALUES ?x0 {%s}.
VALUES ?x3 {%s}
BIND(CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}} UNION {
SELECT DISTINCT ?out WHERE{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
?x2 ns:type.object.type ?x3.
FILTER(?x0 != ?x2).
FILTER(?y2 != ns:type.type.instance)
OPTIONAL{?x2 ns:type.object.name ?name2}.
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.")).
VALUES ?x0 {%s}.
VALUES ?x3 {%s}
BIND(CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}} UNION {
SELECT DISTINCT ?out WHERE{
?x0 ?y1 ?x1.
?x1 ?y4 ?x4.
?x4 ?y2 ?x2.
?x2 ns:type.object.type ?x3.
FILTER(?x0 != ?x2).
FILTER(?y2 != ns:type.type.instance && ?y4 != ns:type.type.instance).
OPTIONAL{?x2 ns:type.object.name ?name2}.
OPTIONAL{?x1 ns:type.object.name ?name1}.
OPTIONAL{?x4 ns:type.object.name ?name4}.
FILTER(BOUND(?name2) && !(BOUND(?name1) && BOUND(?name4)) && regex(str(?x1), "ns/m\\\.|ns/g\\\.") && regex(str(?x4), "ns/m\\\.|ns/g\\\.") && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
VALUES ?x0 {%s}.
VALUES ?x3 {%s}
BIND(CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y4),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}}
}
'''

# when topic entity is relation and another is type
relation_type_sparql = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT * {
{SELECT DISTINCT ?out WHERE{
?x0 ?y2 ?x2.
?x2 ns:type.object.type ?x3.
FILTER(?x0 != ?x2).
OPTIONAL{?x2 ns:type.object.name ?name2}.
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
FILTER(?y2 = %s).
VALUES ?x3 {%s}.
BIND(CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}} UNION {
SELECT DISTINCT ?out WHERE{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
?x2 ns:type.object.type ?x3.
FILTER(?x0 != ?x2).
FILTER(?y2 != ns:type.type.instance)
OPTIONAL{?x2 ns:type.object.name ?name2}.
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.")).
FILTER(?y1 = %s || %s)
VALUES ?x3 {%s}.
BIND(CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}} UNION {
SELECT DISTINCT ?out WHERE{
?x0 ?y1 ?x1.
?x1 ?y4 ?x4.
?x4 ?y2 ?x2.
?x2 ns:type.object.type ?x3.
FILTER(?x0 != ?x2).
FILTER(?y2 != ns:type.type.instance && ?y4 != ns:type.type.instance).
OPTIONAL{?x2 ns:type.object.name ?name2}.
OPTIONAL{?x1 ns:type.object.name ?name1}.
OPTIONAL{?x4 ns:type.object.name ?name4}.
FILTER(?y1 = %s || ?y2 = %s || ?y4 = %s)
FILTER(BOUND(?name2) && !(BOUND(?name1) && BOUND(?name4)) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.") && regex(str(?x4), "ns/m\\\.|ns/g\\\.")).
VALUES ?x3 {%s}.
BIND(CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}}
}
'''

# when topic entity is mid
mid_sparql_hop12 = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT DISTINCT ?out WHERE {
{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
?x2 ns:type.object.name ?name2.
FILTER(?x0 != ?x2).
FILTER(regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.")).
}UNION {
?x0 ?y2 ?x2.
?x2 ns:type.object.name ?name2.
FILTER(?x0 != ?x2).
FILTER(regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
}
VALUES ?x0 {%s}.
BIND(
IF(BOUND(?x1),
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x2)),
CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x2))) AS ?out)
}
'''

mid_sparql_hop3 = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT DISTINCT ?out WHERE {
{
?x0 ?y1 ?x1.
?x1 ?y3 ?x3.
?x3 ?y2 ?x2.
?x2 ns:type.object.name ?name2.
OPTIONAL{?x1 ns:type.object.name ?name1}
OPTIONAL{?x3 ns:type.object.name ?name3}
FILTER(?x0 != ?x2).
FILTER(!(BOUND(?name1) && BOUND(?name3)) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.") && regex(str(?x3), "ns/m\\\.|ns/g\\\.")).
}
VALUES ?x0 {%s}.
BIND(
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y3),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}
'''

mid_sparql_hop4 = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT DISTINCT ?out WHERE {
{
?x0 ?y1 ?x1.
?x1 ?y3 ?x3.
?x3 ?y4 ?x4.
?x4 ?y2 ?x2.
?x2 ns:type.object.name ?name2.
?x3 ns:type.object.name ?name3
OPTIONAL{?x1 ns:type.object.name ?name1}
OPTIONAL{?x4 ns:type.object.name ?name4}
FILTER(?x0 != ?x2).
FILTER(!(BOUND(?name1) && BOUND(?name4)) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.") && regex(str(?x3), "ns/m\\\.|ns/g\\\.")).
}
VALUES ?x0 {%s}.
BIND(
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y3),'\t',str(?y4),'\t',str(?y2),'\t',str(?x2)) AS ?out)
}
'''

# when topic entity is relation
relation_sparql = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT DISTINCT ?out WHERE {
{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
?x2 ns:type.object.name ?name2.
FILTER(?x0 != ?x2).
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.")).
FILTER(?y1 = %s || ?y2 = %s)
} UNION {
?x0 ?y2 ?x2.
?x2 ns:type.object.name ?name2.
FILTER(?x0 != ?x2).
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
FILTER(?y2 = %s)
}
VALUES ?x0 {%s}
BIND(IF(BOUND(?x1),
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x2)),
CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x2))) AS ?out)
}
'''

# when topic entity is type
type_sparql = '''
PREFIX ns:<http://rdf.freebase.com/ns/>
SELECT DISTINCT ?out WHERE {
{
?x0 ?y1 ?x1.
?x1 ?y2 ?x2.
?x0 ns:type.object.type ?t.
?x2 ns:type.object.name ?name2.
FILTER(?x0 != ?x2).
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.") && regex(str(?x1), "ns/m\\\.|ns/g\\\.")).
BIND('1' AS ?z)
} UNION {
?x0 ?y2 ?x2.
?x0 ns:type.object.type ?t.
?x2 ns:type.object.name ?name2.
FILTER(?x0 != ?x2).
FILTER(BOUND(?name2) && regex(str(?x2), "ns/m\\\.|ns/g\\\.")).
}
VALUES ?t {%s}
VALUES ?x0 {%s}
BIND(IF(BOUND(?x1),
CONCAT(str(?x0),'\t',str(?y1),'\t',str(?y2),'\t',str(?x2)),
CONCAT(str(?x0),'\t',str(?y2),'\t',str(?x2))) AS ?out)
}
'''

def mid_query(mid):
    sparql = SPARQLWrapper(SPARQL_PATH)
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

def mid2name_query(mid):
    ss = "ns:%s" %mid
    sparql = SPARQLWrapper(SPARQL_PATH)
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

def type_query(mid):
# %s ns:common.topic.notable_types ?x1.
# ?x1 ns:type.object.name ?x2
    ss = "ns:%s" %mid
    sparql = SPARQLWrapper(SPARQL_PATH)
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

def hop1_rel_query(r):
    h, r = r.split('\t')
    h = "ns:%s" %h
    r = "ns:%s" %r
    sparql = SPARQLWrapper(SPARQL_PATH)
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
    sparql = SPARQLWrapper(SPARQL_PATH)
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
    sparql = SPARQLWrapper(SPARQL_PATH)
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
    sparql = SPARQLWrapper(SPARQL_PATH)
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


sparql_dic = {'mid_mid_sparql': mid_mid_sparql, 'mid_relation_sparql': mid_relation_sparql,
            'mid_type_sparql': mid_type_sparql, 'relation_type_sparql': relation_type_sparql}

def whether_two_mids(topic_entity):
    return True if np.sum([bool(re.search('^[mg]\.', t)) for t in topic_entity]) == 2 else False

def whether_one_mid_one_relation(topic_entity):
    return True if (np.sum([bool(re.search('^[mg]\.', t)) for t in topic_entity]) == 1 and \
        np.sum([bool(re.search('^1', t)) for t in topic_entity]) == 1) else False

def whether_one_mid_one_type(topic_entity):
    return True if np.sum([bool(re.search('^[mg]\.', t)) for t in topic_entity]) == 1 and \
        np.sum([bool(re.search('^2', t)) for t in topic_entity]) == 1 else False

def whether_one_mid(topic_entity):
    return True if len(topic_entity) == 1 and re.search('^[mg]\.', topic_entity[0]) else False

def whether_one_relation(topic_entity):
    return True if len(topic_entity) == 1 and re.search('^1', topic_entity[0]) else False

def whether_one_type(topic_entity):
    return True if len(topic_entity) == 1 and re.search('^2', topic_entity[0]) else False

def single_query(sparql, query):
    done = False
    limit = 5000
    limit_txt = ''
    results = []
    while (not done) and (limit > 400):
        try:
            sparql.setQuery(query + limit_txt)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()["results"]["bindings"]
            done = True
        except:
            limit_txt = ' limit %s' %(limit)
            limit /= 2
            done = False
    return results, limit


def generate_candidate_path_via_sparql_pool(topic_entities, answers, coincident):
    sparql = SPARQLWrapper(SPARQL_PATH) # Change to the path to your database
    is_responses = np.zeros((len(answers)))
    limits = np.zeros((len(answers)))
    response_numbers = []
    path_cands = []
    for t_idx, topic_entity_group in enumerate(topic_entities):
        skip = False
        #topic_entity = ['m.03_dwn']
        stime = time.time()
        #try:
        new_path_cand = {}
        recall = []
        printout = []
        for topic_entity in list(set(topic_entity_group)):
        #if True:
            #topic_entity = topic_entity[1:] # remove this !
            #print(topic_entity)
            topic_entity = [topic_entity]
            #topic_entity = topic_entity_group
            #print(re.sub('\.|\/', '-', '|'.join(sorted(topic_entity))).encode('ascii', 'ignore').decode('ascii'))

            if True:
                topic_entity = [t[1:] if not re.search('^[mg]\.', t) else t for t in topic_entity]
                cand_path, cand_path_score, cand_path_an, cand_path_cont = coincident[t_idx]
                #print(cand_path_score)
                for p_idx, p in enumerate(cand_path):
                    if len(topic_entity) > 0 and set(topic_entity).issubset(set(p.split('\t') + cand_path_cont[p_idx])):
                        new_path_cand[p] = (cand_path_an[p_idx], cand_path_score[p_idx], cand_path_cont[p_idx])

                is_responses[t_idx] = 0
                limits[t_idx] = 0

            elif False:#os.path.exists('data/freebase/cache/%s.pkl' %re.sub('\.|\/', '-', '|'.join(sorted(topic_entity))).encode('ascii', 'ignore').decode('ascii')):
                path_cand, is_response, limit = pickle.load(open('data/freebase/cache2/%s.pkl'
                        %re.sub('\.|\/', '-', '|'.join(sorted(topic_entity))).encode('ascii', 'ignore').decode('ascii'), 'rb'), encoding='latin1')

                for p_idx, p in enumerate(path_cand):
                    if answers[t_idx] == '':
                        f1 = 0
                    else:
                        f1 = compute_f1([a.lower() for a in path_cand[p]], answers[t_idx])
                        # f1 = 0 if f1 < 0.1 else f1
                        #f1 = float(p.lower() in answers[t_idx])
                        printout += ['%s\t%s\t%s' %(p, answers[t_idx], f1)]
                        recall += [f1]
                    if f1 > 0 or np.random.random() < 500./len(path_cand):
                        # try:
                        #     if np.max(recall) == 0:
                        #         print('\n'.join(printout))
                        # except:
                        #     pass
                        print(p)
                        new_path_cand[p] = (len(path_cand[p]), f1)

                is_responses[t_idx] = int(is_response)*int(len(path_cand) > 0)
                limits[t_idx] = limit/5000. if limit != 1 else limit

            else:
                continue
                if len(topic_entity) == 2:
                    if whether_two_mids(topic_entity):
                        results = []
                        topic_entity = ['ns:%s' %t for t in topic_entity]
                        result, limit = single_query(sparql, mid_mid_sparql %(topic_entity[0], topic_entity[1]))
                        results += result
                        result, limit0 = single_query(sparql, mid_mid_sparql %(topic_entity[1], topic_entity[0]))
                        results += result
                        limit = np.min([limit, limit0])

                    elif whether_one_mid_one_relation(topic_entity):
                        topic_entity = ['ns:%s' %t if re.search('^[mg]\.', t) else 'ns:%s' %t[1:] for t in topic_entity]
                        if re.search('^ns:[mg]\.', topic_entity[0]):
                            results, limit = single_query(sparql, mid_relation_sparql %((topic_entity[1],)*2+(topic_entity[0],)+(topic_entity[1],)*3+(topic_entity[0],)))
                        else:
                            results, limit = single_query(sparql, mid_relation_sparql %((topic_entity[0],)*2+(topic_entity[1],)+(topic_entity[0],)*3+(topic_entity[1],)))

                    elif whether_one_mid_one_type(topic_entity):
                        topic_entity = ['ns:%s' %t if re.search('^[mg]\.', t) else 'ns:%s' %t[1:] for t in topic_entity]
                        if re.search('^ns:[mg]\.', topic_entity[0]):
                            results, limit = single_query(sparql, mid_type_sparql %(((topic_entity[0],)+(topic_entity[1],))*3))
                        else:
                            results, limit = single_query(sparql, mid_type_sparql %(((topic_entity[1],)+(topic_entity[0],))*3))

                    else:
                        if re.search('^1', topic_entity[0]):
                            topic_entity = ['ns:%s' %t[1:] for t in topic_entity]
                            results, limit = single_query(sparql, relation_type_sparql %(((topic_entity[0],)+(topic_entity[1],))+((topic_entity[0],)*2+
                                    (topic_entity[1],))+((topic_entity[0],)*3+(topic_entity[1],))))
                        else:
                            results, limit = single_query(sparql, relation_type_sparql %(((topic_entity[1],)+(topic_entity[0],))+((topic_entity[1],)*2+
                                    (topic_entity[0],))+((topic_entity[1],)*3+(topic_entity[0],))))

                elif len(topic_entity) == 1:
                    if whether_one_mid(topic_entity):
                        results = []
                        result, limit = single_query(sparql, mid_sparql_hop12 %('ns:'+topic_entity[0]))
                        results += result
                        #result, limit0 =  single_query(sparql, mid_sparql_hop3 %('ns:'+topic_entity[0]))
                        #results += result
                        limit = limit #np.min([limit, limit0])

                    elif whether_one_relation(topic_entity):
                        cons = ' '.join(['ns:'+ w for w in coincident[topic_entity[0][1:]] if re.search('^[mg]\.', w)])
                        results, limit = single_query(sparql, relation_sparql %(('ns:'+topic_entity[0][1:], )*3+(cons, )))

                    elif whether_one_type(topic_entity):
                        cons = ' '.join(['ns:'+ w for w in coincident[topic_entity[0][1:]] if re.search('^[mg]\.', w)])
                        results, limit = single_query(sparql, type_sparql %('ns:'+topic_entity[0][1:], cons))

                response_time = time.time() - stime
                limits[t_idx] = limit/5000.
                print(len(results))
                # except:
                #     is_responses += [0]
                #     response_times += [-10]
                #     response_numbers += [0]
                #     skip = True

                path_cand = {}
                for idx, result in enumerate(results):
                    path = result["out"]["value"]
                    path = re.sub('http://rdf.freebase.com/ns/', '', path).split('\t')
                    path_rel = '\t'.join(path[:-1])
                    path_ans = path[-1]

                    if (re.search('^[mg]\.', path_ans) or ('index.html?curid' not in path_ans)):
                        path_ans = re.sub('^http://en.wikipedia.org/wiki/', '', path_ans)
                        if path_rel not in path_cand:
                            path_cand[path_rel] = set([path_ans])
                        else:
                            path_cand[path_rel].add(path_ans)
                response_numbers += [len(path_cand)]
                is_responses[t_idx] = int(len(results) > 0)*int(len(path_cand) > 0)

                if response_time > 10:
                    pickle.dump([path_cand, len(path_cand)> 0, limit], open('data/freebase/cache/%s.pkl' %('|'.join(sorted(topic_entity)).replace('.', '-')), 'w'), protocol = 2)

                for p in path_cand:
                    if answers[t_idx] == '':
                        f1 = 0
                    else:
                        f1 = compute_f1(path_cand[p], answers[t_idx].split('|'))
                        f1 = 0 if f1 < 0.1 else f1
                    if f1 > 0 or np.random.random() < 500./len(path_cand):
                        new_path_cand[p] = (len(path_cand[p]), f1)

        path_cands += [new_path_cand]
        # if len(new_path_cand) != len(cand_path):
        #     print(topic_entity_group)
        #     print(cand_path)
        #     exit()
    return path_cands, is_responses, limits

def compute_f1(preds, answers):
	precision = len(set(preds) & set(answers))*1./max([len(set(preds)), 1e-10])
	recall = len(set(preds) & set(answers))*1./max([len(set(answers)), 1e-10])
	f1 = 2*precision*recall/max([(precision + recall), 1e-10])
	return f1
