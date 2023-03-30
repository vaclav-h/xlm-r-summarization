"""
build_ext_dataset.py: Prepares dataset for extractive summarization provided dataset for
                      abstractive summarization in jsonl format with fields: 'text' and 'abstract'
"""


import argparse
import torch
import numpy as np
from rouge_raw import RougeRaw
from datasets import load_dataset
from sentence_splitter import SentenceSplitter


def find_targets(src_sent, tgt_sent, rouge):
    tgt_idx = []
    for tgt in tgt_sent:
        scores = np.zeros(len(src_sent))
        for (i, sent) in enumerate(src_sent):
            rr = rouge.document(tgt, sent)
            scores[i] = (rr['1'].f + rr['2'].f + rr['L'].f) / 3
        
        ranking = np.flip(scores.argsort())
        for i in ranking:
            if int(i) not in tgt_idx:
                tgt_idx.append(int(i))
                break
    return sorted(tgt_idx)


def split_and_extract(example, splitter, rouge, metric):
    src_sent = splitter.split(example['text'])
    tgt_sent = splitter.split(example['abstract'])
    
    tgt_sentences = find_targets(src_sent, tgt_sent, rouge)

    return {'text' : src_sent, 'tgt_sentences' : tgt_sentences}


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        help="path to jsonl file with fields 'text' and 'abstract'")
    parser.add_argument('-o', '--output', type=str,
                        help='output file name')
    parser.add_argument('-l', '--lang', type=str,
                        help='language of data, eg. en, de, cs, ...')
    args = parser.parse_args()
    
    data = load_dataset('json', data_files=args.file)['train']
    splitter = SentenceSplitter(language=args.lang)
    rouge = RougeRaw()
    
    data = data.map(split_and_extract,
                    fn_kwargs={'splitter':splitter, 'rouge':rouge},
                    num_proc=8)
    data.to_json(args.output, force_ascii=False)    
    