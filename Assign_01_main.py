import sys
import pytrec_eval
import json
import nltk
from nltk.corpus import wordnet as wn
import numpy
import heapq

import utils.utils_MED
import utils.utils_top_k

def eval_success(path):
    file_ = open(path, 'r')
    lines = file_.readlines()

    avg_success = [0,0,0]
    qrel = {}
    run = {}

    lst_k = [1, 5, 10]
    largest_k = lst_k[-1]
    word_cor = ""
    for l in lines:
        if '$' in l:
            word_cor = l.split("$")[1].split("\n")[0].lower()
        else:
            word_inc = l.split("\n")[0].lower()
            qrel[word_inc] = {word_cor : 1}
            run[word_inc] = {}
            # Find the k-closest wrods
            top_largest_k_heap = find_top_k_similar(word_inc, largest_k)
            for i in range(1, len(lst_k) + 1):
                cut_off = lst_k[len(lst_k) - i]
                key_val = 1 / cut_off
                interval = cut_off
                if not i == len(lst_k):
                    interval -= lst_k[len(lst_k) - i - 1]
                for j in range(interval):
                    predict = top_largest_k_heap.extractMax()[1]
                    run[word_inc][predict] = key_val
    # Apply evaluation measures
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'success'})
    print(json.dumps(evaluator.evaluate(run), indent=2))
    res = evaluator.evaluate(run)
    print('Averages:')
    idx = 0
    for measure in list(res[list(res.keys())[0]].keys()):
        q = [query_measures[measure] for query_measures in res.values()]
        val = pytrec_eval.compute_aggregated_measure(measure, q)
        avg_success[idx] = val
        idx += 1
        print("  ", measure, val)
    return avg_success



if __name__ == "__main__":
    path = '/content/birkbeck.dat'
    eval_success(path)


