import sys
import json
import numpy as np

def FindInList(entry,elist):
    for item in elist:
        if entry == item:
            return True
    return False

def Hit1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return 1
        else:
            return 0
    elif len(predAnswerList)==0:
        return 0
    else:
        glist =[x["AnswerArgument"] for x in goldAnswerList]
        plist = np.random.choice(predAnswerList, 1)[0]
        return int(plist in glist)

def CalculatePRF1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [0.0, 1.0, 0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList)==0:
        return [1.0, 0.0, 0.0]    # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist =[x["AnswerArgument"] for x in goldAnswerList]
        plist =predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry,plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not FindInList(pentry,glist):
                fp += 1


        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

        f1 = (2*precision*recall)/(precision+recall)
        return [precision, recall, f1]


def main(goldData, predAnswers):
    # if len(sys.argv) != 3:
    #     print ("Usage: python eval.py goldData predAnswers")
    #     sys.exit(-1)
    #
    # goldData = json.loads(open(sys.argv[1]).read())
    # predAnswers = json.loads(open(sys.argv[2]).read())

    goldData = json.loads(open(goldData, 'rb').read())
    predAnswers = json.loads(open(predAnswers).read())

    PredAnswersById = {}

    for item in predAnswers:
        PredAnswersById[item["QuestionId"]] = item["Answers"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    hitSum = 0.0
    numCorrect = 0
    for entry in goldData["Questions"]:

        skip = True
        for pidx in range(0,len(entry["Parses"])):
            np = entry["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(entry["Parses"])==0 or skip):
            continue

        total += 1

        id = entry["QuestionId"]

        if id not in PredAnswersById:
            print ("The problem " + id + " is not in the prediction set")
            print ("Continue to evaluate the other entries")
            continue

        if len(entry["Parses"]) == 0:
            print ("Empty parses in the gold set. Breaking!!")
            break

        predAnswers = PredAnswersById[id]

        bestf1 = -9999
        bestf1Rec = -9999
        bestf1Prec = -9999
        besthit1 = 0
        for pidx in range(0,len(entry["Parses"])):
            pidxAnswers = entry["Parses"][pidx]["Answers"]
            prec,rec,f1 = CalculatePRF1(pidxAnswers,predAnswers)
            hit1 = Hit1(pidxAnswers,predAnswers)
            if f1 > bestf1:
                bestf1 = f1
                bestf1Rec = rec
                bestf1Prec = prec
            if hit1 > besthit1:
                besthit1 = hit1
        # if 0 < bestf1Prec < 1.:
        #     print('%s\t%s\t%s' %(id, entry["RawQuestion"], bestf1))

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        hitSum += besthit1
        if bestf1 == 1.0:
            numCorrect += 1

    #print ("Number of questions:", int(total))
    print ("Hit@1 over questions %.3f" % (hitSum/ total))
    print ("Average precision over questions: %.3f" % (precSum / total))
    print ("Average recall over questions: %.3f" % (recSum / total))
    print ("Average f1 over questions (accuracy): %.3f" % (f1sum / total))
    print ("F1 of average recall and average precision: %.3f" % (2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
    print ("True accuracy (ratio of questions answered exactly correctly): %.3f" % (numCorrect / total))

    return (f1sum / total)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

'''
python code/eval.py  data/WBQ/WebQSP.test.json saved-model/WBQ/final_predictionsTG+EE_tmp.json
'''
