import torch
import collections
import math
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                           translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                             possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def rouge_scores(reference , hypothesis):
    # 计算 ROUGE-1、ROUGE-2 和 ROUGE-L 的得分
    # rouge_1_score = sentence_bleu([reference], hypothesis, weights=(1, 0, 0), smoothing_function=SmoothingFunction().method4)
    # rouge_2_score = sentence_bleu([reference], hypothesis, weights=(0, 1, 0), smoothing_function=SmoothingFunction().method4)
    rouge_l_score = sentence_bleu([reference], hypothesis, weights=(0, 0, 1), smoothing_function=SmoothingFunction().method4)
    
    return rouge_l_score


def bleu_score(candidate, reference):
    # 计算 BLEU 分数
    score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method2)
    return score


def calc_length(path, key='instruction'):
    dataset = datasets.Dataset.from_json(path)
    avg_len = 0
    for sample in dataset:
        avg_len += len(sample[key])

    avg_len /= len(dataset)
    print(avg_len)
    return avg_len
