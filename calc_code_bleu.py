# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import os
import json
import pandas as pd

from codegen.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen.preprocessing.lang_processors.python_processor import PythonProcessor
from evaluation.CodeBLEU import (
    bleu,
    weighted_ngram_match,
    syntax_match,
    dataflow_match
)

pyprocessor = PythonProcessor()
jprocessor = JavaProcessor("./third_party")

def language_specific_processing(tokens, lang):
    if lang == 'python':
        return pyprocessor.detokenize_code(' '.join(tokens)).split()
    else:
        return jprocessor.detokenize_code(' '.join(tokens)).split()

def get_codebleu(
        ref,
        hyp,
        lang,
        params='0.25,0.25,0.25,0.25',
        txt_ref=False,
        keyword_dir=None
):
    lang = 'javascript' if lang == 'js' else lang
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]

    df = pd.read_csv(ref, header=None)
    references = [[d] for d in df[0].tolist()]
    df = pd.read_csv(hyp, header=None)
    hypothesis = df[0].tolist()

    assert len(hypothesis) == len(references)

    # calculate ngram match (BLEU)
    tokenized_hyps = [language_specific_processing(str(x).split(), lang) for x in hypothesis]
    tokenized_refs = [[language_specific_processing(str(x).split(), lang) for x in reference] for reference in references]

    count = 0
    for i in range(len(tokenized_refs)):
        refs = tokenized_refs[i]  # r is a list of 'list of tokens'
        # print(refs)
        t = tokenized_hyps[i]  # 'list of tokens'
        # print(t)
        for r in refs:
            if r == t:
                count += 1
                break
    em = round(count / len(tokenized_refs) * 100, 2)

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    if keyword_dir is None:
        keyword_dir = "evaluation/CodeBLEU/keywords"

    kw_file = os.path.join(keyword_dir, '{}.txt'.format(lang))
    keywords = [x.strip() for x in open(kw_file, 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [
        [
            [reference_tokens, make_weights(reference_tokens, keywords)] for reference_tokens in reference
        ] for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    print(
        'EM:\t%.2f\nNgram match:\t%.2f\nWeighted ngram:\t%.2f\nSyntax match:\t%.2f\nDataflow match:\t%.2f' %
        (em, ngram_match_score * 100, weighted_ngram_match_score * 100,
         syntax_match_score * 100, dataflow_match_score * 100)
    )

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    return code_bleu_score


if __name__ == '__main__':
    code_bleu_score = get_codebleu('result/defense/python-to-java/gold.csv',
                                   'result/defense_AT_original/python-to-java/plbart.csv', 'java', '0.25,0.25,0.25,0.25', True)

    code_bleu_score = get_codebleu('result/defense/python-to-java/gold.csv',
                                   'result/defense_AT/python-to-java/plbart.csv', 'java', '0.25,0.25,0.25,0.25', True)

    # code_bleu_score = get_codebleu('result/defense/python-to-java/gold.csv',
    #                                'result/defense/python-to-java/codegen.csv', 'java', '0.25,0.25,0.25,0.25', True)

    # code_bleu_score = get_codebleu('result/python-to-java/gold.csv',
    #                                'defense/random_1/syntax-attack-python-to-java/natgen.csv', 'java', '0.25,0.25,0.25,0.25', True)
    # print('CodeBLEU score: %.2f' % (code_bleu_score * 100.0))