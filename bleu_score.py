from nltk.translate.bleu_score import corpus_bleu

from settings import *


# 多个句子
def read_references():
    result = []
    f = open(BLEU_REFERENCES, 'r', encoding='utf-8')
    sentences = f.readlines()
    for s in sentences:
        references = []
        references.append(s.strip().split(' '))
        result.append(references)
    f.close()
    return result


def read_candidates():
    result = []
    file = open(BLEU_CANDIDATE, 'r', encoding='utf-8')
    sentences = file.readlines()
    for s in sentences:
        result.append(s.strip().split(' '))
    file.close()
    return result


if __name__ == '__main__':
    references = read_references()
    candidates = read_candidates()
    score = corpus_bleu(references, candidates, weights=(1, 0.2, 0, 0))
    print(score)
