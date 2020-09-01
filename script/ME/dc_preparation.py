#!/usr/bin/env python
# coding: utf-8

import argparse
import re
import os
import _pickle as cPickle
import numpy as np
import pandas as pd
import torch
from pytorch_transformers.tokenization_bert import BertTokenizer

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)
    
# the same tokenize function from BERT adapted for this task
def tokenize(entries, tokenizer, max_length=16, padding_index=0):
    """Tokenizes the captions.

    This will add c_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_index in embedding
    """
    for entry in entries:
        tokens = tokenizer.encode(entry["caption"])
        tokens = tokens[: max_length - 2]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [padding_index] * (max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        assert_eq(len(tokens), max_length)
        entry["c_token"] = tokens
        entry["c_input_mask"] = input_mask
        entry["c_segment_ids"] = segment_ids

# the same tensorize function from BERT adapted for this task
def tensorize(entries, split='trainval'):

    for entry in entries:
        caption = torch.from_numpy(np.array(entry["c_token"]))
        entry["c_token"] = caption

        c_input_mask = torch.from_numpy(np.array(entry["c_input_mask"]))
        entry["c_input_mask"] = c_input_mask

        c_segment_ids = torch.from_numpy(np.array(entry["c_segment_ids"]))
        entry["c_segment_ids"] = c_segment_ids

        if "scores" in entry:
            scores = np.array(entry["scores"], dtype=np.float32)
            scores = torch.from_numpy(scores)
            entry["scores"] = scores

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--captions_path",
        type=str, 
        default="/MediaEval/alto_titles_danny.csv", 
        help="Captions .csv file"
    )
    
    parser.add_argument(
        "--train_gt",
        type=str, 
        default="/MediaEval/dev-set/ground-truth/ground-truth_dev-set.csv", 
        help="Ground truth .csv file for the training set"
    )
    
    parser.add_argument(
        "--test_gt",
        type=str, 
        default="/MediaEval/test-set/ground-truth/ground-truth_test-set.csv", 
        help="Ground truth .csv file for the training set"
    )
    
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    
    parser.add_argument(
        "--split",
        default="trainval", 
        type=str, 
        help="which split to use (trainval or test or trainval-test). Default is trainval"
    )
    
    args = parser.parse_args()

    #deep_coptions_path = "/MediaEval/alto_titles_danny.csv"
    
    dataroot = 'datasets/ME'
    max_length = 23
    deep_coptions_df = pd.read_csv(args.captions_path)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    dc_entries = []
    for r in deep_coptions_df.itertuples():
        sample = {}
        vid_id = int(r.video)
        caption = r.caption.rstrip().replace('-', ' ')
        sample['video_id'] = vid_id
        sample['caption'] = caption
        dc_entries.append(sample)

        
    
    if "train" in args.split:
        train_df = pd.read_csv(args.train_gt)
        score_dict = {}
        for r in train_df.itertuples():
            vid_id = re.findall(r'\d+', r.video)[0]
            vid_id = int(vid_id)
            score_dict[vid_id] = [r._2, r._4]
        
        train_score_list = []
        for sample in dc_entries:
            if sample['video_id'] in score_dict:
                sample['scores'] = score_dict[sample['video_id']]
                train_score_list.append(sample)
        
        tokenize(train_score_list, tokenizer, max_length=max_length)
        tensorize(train_score_list, split="trainval")
        train_cache_path = os.path.join(dataroot, 'cache', 'ME' + '_' + "trainval" + '_' + str(max_length) + '_cleaned' + '.pkl')
        print("Saving cache file with {} samples under {}".format(len(train_score_list), train_cache_path))
        cPickle.dump(train_score_list, open(train_cache_path, 'wb'))
    
    
    if "test" in args.split:
        test_df = pd.read_csv(args.test_gt)
        test_score_dict = {}
        for r in test_df.itertuples():
            vid_id = re.findall(r'\d+', r.video)[0]
            vid_id = int(vid_id)
            test_score_dict[vid_id] = [r._2, r._4]
        
        test_score_list = []
        for sample in dc_entries:
            if sample['video_id'] in test_score_dict:
                sample['scores'] = test_score_dict[sample['video_id']]
                test_score_list.append(sample)
                
        tokenize(test_score_list, tokenizer, max_length=max_length)
        tensorize(test_score_list, split="test")
        test_cache_path = os.path.join(dataroot, 'cache', 'ME' + '_' + "test" + '_' + str(max_length) + '_cleaned' + '.pkl')
        print("Saving cache file with {} samples under {}".format(len(test_score_list), test_cache_path))
        cPickle.dump(test_score_list, open(test_cache_path, 'wb'))


if __name__ == "__main__":

    main()
